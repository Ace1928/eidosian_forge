import collections
import contextlib
import os
import re
import threading
import time
import weakref
from six.moves import queue
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.distribute.coordinator import metric_utils
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.distribute.coordinator import utils
from tensorflow.python.distribute.coordinator import values as values_lib
from tensorflow.python.distribute.coordinator import watchdog
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@tf_export('distribute.experimental.coordinator.ClusterCoordinator', 'distribute.coordinator.ClusterCoordinator', v1=[])
class ClusterCoordinator(object):
    """An object to schedule and coordinate remote function execution.

  This class is used to create fault-tolerant resources and dispatch functions
  to remote TensorFlow servers.

  Currently, this class is not supported to be used in a standalone manner. It
  should be used in conjunction with a `tf.distribute` strategy that is designed
  to work with it. The `ClusterCoordinator` class currently only works
  `tf.distribute.experimental.ParameterServerStrategy`.

  __The `schedule`/`join` APIs__

  The most important APIs provided by this class is the `schedule`/`join` pair.
  The `schedule` API is non-blocking in that it queues a `tf.function` and
  returns a `RemoteValue` immediately. The queued functions will be dispatched
  to remote workers in background threads and their `RemoteValue`s will be
  filled asynchronously. Since `schedule` doesn’t require worker assignment, the
  `tf.function` passed in can be executed on any available worker. If the worker
  it is executed on becomes unavailable before its completion, it will be
  migrated to another worker. Because of this fact and function execution is not
  atomic, a function may be executed more than once.

  __Handling Task Failure__

  This class when used with
  `tf.distribute.experimental.ParameterServerStrategy`, comes with built-in
  fault tolerance for worker failures. That is, when some workers are not
  available for any reason to be reached from the coordinator, the training
  progress continues to be made with the remaining workers. Upon recovery of a
  failed worker, it will be added for function execution after datasets created
  by `create_per_worker_dataset` are re-built on it.

  When a parameter server fails, a `tf.errors.UnavailableError` is raised by
  `schedule`, `join` or `done`. In this case, in addition to bringing back the
  failed parameter server, users should restart the coordinator so that it
  reconnects to workers and parameter servers, re-creates the variables, and
  loads checkpoints. If the coordinator fails, after the user brings it back,
  the program will automatically connect to workers and parameter servers, and
  continue the progress from a checkpoint.

  It is thus essential that in user's program, a checkpoint file is periodically
  saved, and restored at the start of the program. If an
  `tf.keras.optimizers.Optimizer` is checkpointed, after restoring from a
  checkpoiont, its `iterations` property roughly indicates the number of steps
  that have been made. This can be used to decide how many epochs and steps are
  needed before the training completion.

  See `tf.distribute.experimental.ParameterServerStrategy` docstring for an
  example usage of this API.

  This is currently under development, and the API as well as implementation
  are subject to changes.
  """

    def __new__(cls, strategy):
        if strategy._cluster_coordinator is None:
            strategy._cluster_coordinator = super(ClusterCoordinator, cls).__new__(cls)
        return strategy._cluster_coordinator

    def __init__(self, strategy):
        """Initialization of a `ClusterCoordinator` instance.

    Args:
      strategy: a supported `tf.distribute.Strategy` object. Currently, only
        `tf.distribute.experimental.ParameterServerStrategy` is supported.

    Raises:
      ValueError: if the strategy being used is not supported.
    """
        if not getattr(self, '_has_initialized', False):
            if not hasattr(strategy, '_is_parameter_server_strategy_v2'):
                raise ValueError('Only `tf.distribute.experimental.ParameterServerStrategy` is supported to work with `tf.distribute.experimental.coordinator.ClusterCoordinator` currently.')
            self._strategy = strategy
            self.strategy.extended._used_with_coordinator = True
            self._cluster = Cluster(strategy)
            self._has_initialized = True

    def __del__(self):
        logging.info('ClusterCoordinator destructor: stopping cluster')
        self._cluster.stop()

    @property
    def strategy(self):
        """Returns the `Strategy` associated with the `ClusterCoordinator`."""
        return self._strategy

    def schedule(self, fn, args=None, kwargs=None):
        """Schedules `fn` to be dispatched to a worker for asynchronous execution.

    This method is non-blocking in that it queues the `fn` which will be
    executed later and returns a
    `tf.distribute.experimental.coordinator.RemoteValue` object immediately.
    `fetch` can be called on it to wait for the function execution to finish
    and retrieve its output from a remote worker. On the other hand, call
    `tf.distribute.experimental.coordinator.ClusterCoordinator.join` to wait for
    all scheduled functions to finish.

    `schedule` guarantees that `fn` will be executed on a worker at least once;
    it could be more than once if its corresponding worker fails in the middle
    of its execution. Note that since worker can fail at any point when
    executing the function, it is possible that the function is partially
    executed, but `tf.distribute.experimental.coordinator.ClusterCoordinator`
    guarantees that in those events, the function will eventually be executed on
    any worker that is available.

    If any previously scheduled function raises an error, `schedule` will raise
    any one of those errors, and clear the errors collected so far. What happens
    here, some of the previously scheduled functions may have not been executed.
    User can call `fetch` on the returned
    `tf.distribute.experimental.coordinator.RemoteValue` to inspect if they have
    executed, failed, or cancelled, and reschedule the corresponding function if
    needed.

    When `schedule` raises, it guarantees that there is no function that is
    still being executed.

    At this time, there is no support of worker assignment for function
    execution, or priority of the workers.

    `args` and `kwargs` are the arguments passed into `fn`, when `fn` is
    executed on a worker. They can be
    `tf.distribute.experimental.coordinator.PerWorkerValues` and in this case,
    the argument will be substituted with the corresponding component on the
    target worker. Arguments that are not
    `tf.distribute.experimental.coordinator.PerWorkerValues` will be passed into
    `fn` as-is. Currently, `tf.distribute.experimental.coordinator.RemoteValue`
    is not supported to be input `args` or `kwargs`.

    Args:
      fn: A `tf.function`; the function to be dispatched to a worker for
        execution asynchronously. Regular python function is not supported to be
        scheduled.
      args: Positional arguments for `fn`.
      kwargs: Keyword arguments for `fn`.

    Returns:
      A `tf.distribute.experimental.coordinator.RemoteValue` object that
      represents the output of the function scheduled.

    Raises:
      Exception: one of the exceptions caught by the coordinator from any
        previously scheduled function, since the last time an error was thrown
        or since the beginning of the program.
    """
        if not isinstance(fn, (def_function.Function, tf_function.ConcreteFunction)):
            raise TypeError('`tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` only accepts a `tf.function` or a concrete function.')
        with self.strategy.scope():
            self.strategy.extended._being_scheduled = True
            schedule_remote_value = self._cluster.schedule(fn, args=args, kwargs=kwargs)
            self.strategy.extended._being_scheduled = False
            return schedule_remote_value

    def join(self):
        """Blocks until all the scheduled functions have finished execution.

    If any previously scheduled function raises an error, `join` will fail by
    raising any one of those errors, and clear the errors collected so far. If
    this happens, some of the previously scheduled functions may have not been
    executed. Users can call `fetch` on the returned
    `tf.distribute.experimental.coordinator.RemoteValue` to inspect if they have
    executed, failed, or cancelled. If some that have been cancelled need to be
    rescheduled, users should call `schedule` with the function again.

    When `join` returns or raises, it guarantees that there is no function that
    is still being executed.

    Raises:
      Exception: one of the exceptions caught by the coordinator by any
        previously scheduled function since the last time an error was thrown or
        since the beginning of the program.
    """
        self._cluster.join()

    def done(self):
        """Returns whether all the scheduled functions have finished execution.

    If any previously scheduled function raises an error, `done` will fail by
    raising any one of those errors.

    When `done` returns True or raises, it guarantees that there is no function
    that is still being executed.

    Returns:
      Whether all the scheduled functions have finished execution.
    Raises:
      Exception: one of the exceptions caught by the coordinator by any
        previously scheduled function since the last time an error was thrown or
        since the beginning of the program.
    """
        return self._cluster.done()

    def create_per_worker_dataset(self, dataset_fn):
        """Create dataset on each worker.

    This creates dataset on workers from the input which can be either a
    `tf.data.Dataset`, a `tf.distribute.DistributedDataset` or a function which
    returns a dataset, and returns an object that represents the collection of
    those individual datasets. Calling `iter` on such collection of datasets
    returns a `tf.distribute.experimental.coordinator.PerWorkerValues`, which is
    a collection of iterators, where the iterators have been placed on
    respective workers.

    Calling `next` on a `PerWorkerValues` of iterator is unsupported. The
    iterator is meant to be passed as an argument into
    `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule`. When
    the scheduled function is about to be executed by a worker, the
    function will receive the individual iterator that corresponds to the
    worker. The `next` method can be called on an iterator inside a
    scheduled function when the iterator is an input of the function.

    Currently the `schedule` method assumes workers are all the same and thus
    assumes the datasets on different workers are the same, except they may be
    shuffled differently if they contain a `dataset.shuffle` operation and a
    random seed is not set. Because of this, we also recommend the datasets to
    be repeated indefinitely and schedule a finite number of steps instead of
    relying on the `OutOfRangeError` from a dataset.


    Example:

    ```python
    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver=...)
    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
        strategy=strategy)

    @tf.function
    def worker_fn(iterator):
      return next(iterator)

    def per_worker_dataset_fn():
      return strategy.distribute_datasets_from_function(
          lambda x: tf.data.Dataset.from_tensor_slices([3] * 3))

    per_worker_dataset = coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)
    per_worker_iter = iter(per_worker_dataset)
    remote_value = coordinator.schedule(worker_fn, args=(per_worker_iter,))
    assert remote_value.fetch() == 3
    ```

    Args:
      dataset_fn: The dataset function that returns a dataset. This is to be
        executed on the workers.

    Returns:
      An object that represents the collection of those individual
      datasets. `iter` is expected to be called on this object that returns
      a `tf.distribute.experimental.coordinator.PerWorkerValues` of the
      iterators (that are on the workers).
    """
        return values_lib.get_per_worker_dataset(dataset_fn, self)

    def _create_per_worker_resources(self, fn, args=None, kwargs=None):
        """Synchronously create resources on the workers.

    The resources are represented by
    `tf.distribute.experimental.coordinator.RemoteValue`s.

    Args:
      fn: The function to be dispatched to all workers for execution
        asynchronously.
      args: Positional arguments for `fn`.
      kwargs: Keyword arguments for `fn`.

    Returns:
      A `tf.distribute.experimental.coordinator.PerWorkerValues` object, which
      wraps a tuple of `tf.distribute.experimental.coordinator.RemoteValue`
      objects.
    """
        results = []
        for w in self._cluster.workers:
            results.append(w.create_resource(fn, args=args, kwargs=kwargs))
        return PerWorkerValues(tuple(results))

    def _create_per_worker_variables(self, fn, args=None, kwargs=None):
        """Asynchronously create variables on workers."""
        results = []
        for w in self._cluster.workers:
            results.append(w.create_variable_resource(fn, args=args, kwargs=kwargs))
        return PerWorkerValues(tuple(results))

    def fetch(self, val):
        """Blocking call to fetch results from the remote values.

    This is a wrapper around
    `tf.distribute.experimental.coordinator.RemoteValue.fetch` for a
    `RemoteValue` structure; it returns the execution results of
    `RemoteValue`s. If not ready, wait for them while blocking the caller.

    Example:
    ```python
    strategy = ...
    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
        strategy)

    def dataset_fn():
      return tf.data.Dataset.from_tensor_slices([1, 1, 1])

    with strategy.scope():
      v = tf.Variable(initial_value=0)

    @tf.function
    def worker_fn(iterator):
      def replica_fn(x):
        v.assign_add(x)
        return v.read_value()
      return strategy.run(replica_fn, args=(next(iterator),))

    distributed_dataset = coordinator.create_per_worker_dataset(dataset_fn)
    distributed_iterator = iter(distributed_dataset)
    result = coordinator.schedule(worker_fn, args=(distributed_iterator,))
    assert coordinator.fetch(result) == 1
    ```

    Args:
      val: The value to fetch the results from. If this is structure of
        `tf.distribute.experimental.coordinator.RemoteValue`, `fetch()` will be
        called on the individual
        `tf.distribute.experimental.coordinator.RemoteValue` to get the result.

    Returns:
      If `val` is a `tf.distribute.experimental.coordinator.RemoteValue` or a
      structure of `tf.distribute.experimental.coordinator.RemoteValue`s,
      return the fetched `tf.distribute.experimental.coordinator.RemoteValue`
      values immediately if they are available, or block the call until they are
      available, and return the fetched
      `tf.distribute.experimental.coordinator.RemoteValue` values with the same
      structure. If `val` is other types, return it as-is.
    """

        def _maybe_fetch(val):
            if isinstance(val, RemoteValue):
                return val.fetch()
            else:
                return val
        return nest.map_structure(_maybe_fetch, val)