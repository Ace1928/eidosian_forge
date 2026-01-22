import collections
import contextlib
import copy
import enum  # pylint: disable=g-bad-import-order
import functools
import threading
import weakref
import six
from tensorflow.python import tf2
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import numpy_dataset
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context as eager_context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import tape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import tf_logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import distribute as ds_types
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls
@tf_export(v1=['distribute.StrategyExtended'])
class StrategyExtendedV1(StrategyExtendedV2):
    __doc__ = StrategyExtendedV2.__doc__

    def experimental_make_numpy_dataset(self, numpy_input, session=None):
        """Makes a dataset for input provided via a numpy array.

    This avoids adding `numpy_input` as a large constant in the graph,
    and copies the data to the machine or machines that will be processing
    the input.

    Args:
      numpy_input: A nest of NumPy input arrays that will be distributed evenly
        across all replicas. Note that lists of Numpy arrays are stacked, as
        that is normal `tf.data.Dataset` behavior.
      session: (TensorFlow v1.x graph execution only) A session used for
        initialization.

    Returns:
      A `tf.data.Dataset` representing `numpy_input`.
    """
        _require_cross_replica_or_default_context_extended(self)
        return self._experimental_make_numpy_dataset(numpy_input, session=session)

    def _experimental_make_numpy_dataset(self, numpy_input, session):
        raise NotImplementedError('must be implemented in descendants')

    def broadcast_to(self, tensor, destinations):
        """Mirror a tensor on one device to all worker devices.

    Args:
      tensor: A Tensor value to broadcast.
      destinations: A mirrored variable or device string specifying the
        destination devices to copy `tensor` to.

    Returns:
      A value mirrored to `destinations` devices.
    """
        assert destinations is not None
        _require_cross_replica_or_default_context_extended(self)
        assert not isinstance(destinations, (list, tuple))
        return self._broadcast_to(tensor, destinations)

    def _broadcast_to(self, tensor, destinations):
        raise NotImplementedError('must be implemented in descendants')

    @deprecated(None, 'please use `run` instead.')
    def experimental_run_steps_on_iterator(self, fn, iterator, iterations=1, initial_loop_values=None):
        """DEPRECATED: please use `run` instead.

    Run `fn` with input from `iterator` for `iterations` times.

    This method can be used to run a step function for training a number of
    times using input from a dataset.

    Args:
      fn: function to run using this distribution strategy. The function must
        have the following signature: `def fn(context, inputs)`. `context` is an
          instance of `MultiStepContext` that will be passed when `fn` is run.
          `context` can be used to specify the outputs to be returned from `fn`
          by calling `context.set_last_step_output`. It can also be used to
          capture non tensor outputs by `context.set_non_tensor_output`. See
          `MultiStepContext` documentation for more information. `inputs` will
          have same type/structure as `iterator.get_next()`. Typically, `fn`
          will use `call_for_each_replica` method of the strategy to distribute
          the computation over multiple replicas.
      iterator: Iterator of a dataset that represents the input for `fn`. The
        caller is responsible for initializing the iterator as needed.
      iterations: (Optional) Number of iterations that `fn` should be run.
        Defaults to 1.
      initial_loop_values: (Optional) Initial values to be passed into the
        loop that runs `fn`. Defaults to `None`. # TODO(priyag): Remove
          initial_loop_values argument when we have a mechanism to infer the
          outputs of `fn`.

    Returns:
      Returns the `MultiStepContext` object which has the following properties,
      among other things:
        - run_op: An op that runs `fn` `iterations` times.
        - last_step_outputs: A dictionary containing tensors set using
        `context.set_last_step_output`. Evaluating this returns the value of
        the tensors after the last iteration.
        - non_tensor_outputs: A dictionary containing anything that was set by
          `fn` by calling `context.set_non_tensor_output`.
    """
        _require_cross_replica_or_default_context_extended(self)
        with self._container_strategy().scope():
            return self._experimental_run_steps_on_iterator(fn, iterator, iterations, initial_loop_values)

    def _experimental_run_steps_on_iterator(self, fn, iterator, iterations, initial_loop_values):
        raise NotImplementedError('must be implemented in descendants')

    def call_for_each_replica(self, fn, args=(), kwargs=None):
        """Run `fn` once per replica.

    `fn` may call `tf.get_replica_context()` to access methods such as
    `replica_id_in_sync_group` and `merge_call()`.

    `merge_call()` is used to communicate between the replicas and
    re-enter the cross-replica context. All replicas pause their execution
    having encountered a `merge_call()` call. After that the
    `merge_fn`-function is executed. Its results are then unwrapped and
    given back to each replica call. After that execution resumes until
    `fn` is complete or encounters another `merge_call()`.  Example:

    ```python
    # Called once in "cross-replica" context.
    def merge_fn(distribution, three_plus_replica_id):
      # sum the values across replicas
      return sum(distribution.experimental_local_results(three_plus_replica_id))

    # Called once per replica in `distribution`, in a "replica" context.
    def fn(three):
      replica_ctx = tf.get_replica_context()
      v = three + replica_ctx.replica_id_in_sync_group
      # Computes the sum of the `v` values across all replicas.
      s = replica_ctx.merge_call(merge_fn, args=(v,))
      return s + v

    with distribution.scope():
      # in "cross-replica" context
      ...
      merged_results = distribution.run(fn, args=[3])
      # merged_results has the values from every replica execution of `fn`.
      # This statement prints a list:
      print(distribution.experimental_local_results(merged_results))
    ```

    Args:
      fn: function to run (will be run once per replica).
      args: Tuple or list with positional arguments for `fn`.
      kwargs: Dict with keyword arguments for `fn`.

    Returns:
      Merged return value of `fn` across all replicas.
    """
        _require_cross_replica_or_default_context_extended(self)
        if kwargs is None:
            kwargs = {}
        with self._container_strategy().scope():
            return self._call_for_each_replica(fn, args, kwargs)

    def _call_for_each_replica(self, fn, args, kwargs):
        raise NotImplementedError('must be implemented in descendants')

    def read_var(self, v):
        """Reads the value of a variable.

    Returns the aggregate value of a replica-local variable, or the
    (read-only) value of any other variable.

    Args:
      v: A variable allocated within the scope of this `tf.distribute.Strategy`.

    Returns:
      A tensor representing the value of `v`, aggregated across replicas if
      necessary.
    """
        raise NotImplementedError('must be implemented in descendants')

    def update_non_slot(self, colocate_with, fn, args=(), kwargs=None, group=True):
        """Runs `fn(*args, **kwargs)` on `colocate_with` devices.

    Used to update non-slot variables.

    DEPRECATED: TF 1.x ONLY.

    Args:
      colocate_with: Devices returned by `non_slot_devices()`.
      fn: Function to execute.
      args: Tuple or list. Positional arguments to pass to `fn()`.
      kwargs: Dict with keyword arguments to pass to `fn()`.
      group: Boolean. Defaults to True. If False, the return value will be
        unwrapped.

    Returns:
      Return value of `fn`, possibly merged across devices.
    """
        _require_cross_replica_or_default_context_extended(self)
        if kwargs is None:
            kwargs = {}
        fn = autograph.tf_convert(fn, autograph_ctx.control_status_ctx(), convert_by_default=False)
        with self._container_strategy().scope():
            return self._update_non_slot(colocate_with, fn, args, kwargs, group)

    def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
        raise NotImplementedError('must be implemented in descendants')

    def non_slot_devices(self, var_list):
        """Device(s) for non-slot variables.

    DEPRECATED: TF 1.x ONLY.

    This method returns non-slot devices where non-slot variables are placed.
    Users can create non-slot variables on these devices by using a block:

    ```python
    with tf.distribute.StrategyExtended.colocate_vars_with(tf.distribute.StrategyExtended.non_slot_devices(...)):
      ...
    ```

    Args:
      var_list: The list of variables being optimized, needed with the
        default `tf.distribute.Strategy`.
    Returns:
      A sequence of devices for non-slot variables.
    """
        raise NotImplementedError('must be implemented in descendants')

    def _use_merge_call(self):
        """Whether to use merge-calls inside the distributed strategy."""
        return True

    @property
    def experimental_between_graph(self):
        """Whether the strategy uses between-graph replication or not.

      This is expected to return a constant value that will not be changed
      throughout its life cycle.
    """
        raise NotImplementedError('must be implemented in descendants')

    @property
    def experimental_should_init(self):
        """Whether initialization is needed."""
        raise NotImplementedError('must be implemented in descendants')

    @property
    def should_checkpoint(self):
        """Whether checkpointing is needed."""
        raise NotImplementedError('must be implemented in descendants')

    @property
    def should_save_summary(self):
        """Whether saving summaries is needed."""
        raise NotImplementedError('must be implemented in descendants')