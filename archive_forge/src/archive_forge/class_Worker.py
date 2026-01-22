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
class Worker(object):
    """A worker in a cluster.

  Attributes:
    worker_index: The index of the worker in the cluster.
    device_name: The device string of the worker, e.g. "/job:worker/task:1".
    executor: The worker's executor for remote function execution.
    failure_handler: The failure handler used to handler worker preemption
      failure.
  """

    def __init__(self, worker_index, device_name, cluster):
        self.worker_index = worker_index
        self.device_name = device_name
        self.executor = executor.new_executor(enable_async=False)
        self.failure_handler = cluster.failure_handler
        self._cluster = cluster
        self._resource_tracking_lock = threading.Lock()
        self._resource_remote_value_refs = []
        self._is_dead_with_error = None
        self._should_worker_thread_run = True
        threading.Thread(target=self._process_queue, name='WorkerClosureProcessingLoop-%d' % self.worker_index, daemon=True).start()

    def stop(self):
        """Ensure the worker thread is closed."""
        self._should_worker_thread_run = False

    def _schedule_resource(self, closure):
        self._cluster.closure_queue.put(closure, tag=self.worker_index)

    def _set_resources_aborted(self, e):
        """Set the resource ABORTED and add an error to it."""
        logging.info('[Worker %d] Clearing all resources.', self.worker_index)
        for weakref_resource in self._resource_remote_value_refs:
            resource = weakref_resource()
            if resource:
                resource._set_aborted(ClosureAbortedError(e))

    def _on_closure_failure(self, closure, e):
        logging.info('[Worker %d] Putting back a closure after it failed.', self.worker_index)
        self._cluster.closure_queue.put_back(closure)
        with self._resource_tracking_lock:
            self._is_dead_with_error = e
            self._set_resources_aborted(e)

    def _on_resource_closure_failure(self, e):
        """Clear tagged queue to ensure resource closures are rebuilt.

    Args:
      e: The exception arisen from the resource closure.
    """
        logging.info('[Worker %d] Clearing tagged queue after resource closure failure.', self.worker_index)
        with self._resource_tracking_lock:
            self._is_dead_with_error = e
            self._cluster.closure_queue.clear_tag_unlocked(self.worker_index)
            self._set_resources_aborted(e)

    def _on_worker_recovery(self):
        logging.info('[Worker %d] calling _on_worker_recovery', self.worker_index)
        with self._resource_tracking_lock:
            for weakref_resource in self._resource_remote_value_refs:
                resource = weakref_resource()
                if resource:
                    self._schedule_resource(resource._closure)
            self._is_dead_with_error = False

    def _process_closure(self, closure):
        """Runs a closure with preemption handling."""
        try:
            with self.failure_handler.wait_on_failure(on_failure_fn=lambda e: self._on_closure_failure(closure, e), on_transient_failure_fn=lambda: self._cluster.closure_queue.put_back(closure), on_recovery_fn=self._on_worker_recovery, worker_device_name=self.device_name):
                closure.execute_on(self)
                with metric_utils.monitored_timer('remote_value_fetch'):
                    closure.maybe_call_with_output_remote_value(lambda r: r.get())
                self._cluster.closure_queue.mark_finished()
        except Exception as e:
            if not isinstance(e, errors.CancelledError):
                logging.error(' /job:worker/task:%d encountered the following error when processing closure: %r:%s', self.worker_index, e, e)
            closure.maybe_call_with_output_remote_value(lambda r: r._set_error(e))
            self._cluster.closure_queue.mark_failed(e)

    def _process_resource_closure(self, closure):
        """Run the given resource closure with preemption handling."""
        assert closure.tag == self.worker_index
        try:
            with self.failure_handler.wait_on_failure(on_failure_fn=self._on_resource_closure_failure, on_transient_failure_fn=lambda: self._process_resource_closure(closure), on_recovery_fn=self._on_worker_recovery, worker_device_name=self.device_name):
                closure.execute_on(self)
        except Exception as e:
            logging.info('[Worker %d] got an exception when processing resource closure', self.worker_index)
            if not isinstance(e, errors.CancelledError):
                logging.error(' /job:worker/task:%d encountered the following error when processing resource closure: %r:%s', self.worker_index, e, e)
            closure.maybe_call_with_output_remote_value(lambda r: r._set_error(e))

    def _maybe_delay(self):
        """Delay if corresponding env vars are set."""
        delay_secs = int(os.environ.get('TF_COORDINATOR_SCHEDULE_START_DELAY', '0'))
        delay_secs *= self.worker_index
        delay_cap = int(os.environ.get('TF_COORDINATOR_SCHEDULE_START_DELAY_MAX', '0'))
        if delay_cap:
            delay_secs = min(delay_secs, delay_cap)
        if delay_secs > 0:
            logging.info(' Worker %d sleeping for %d seconds before running function', self.worker_index, delay_secs)
        time.sleep(delay_secs)

    def _process_queue(self):
        """Function running in a worker thread to process closure queues."""
        self._maybe_delay()
        while self._should_worker_thread_run:
            closure = self._cluster.closure_queue.get(tag=self.worker_index)
            if not self._should_worker_thread_run or closure is None:
                if closure is not None:
                    closure.mark_cancelled()
                return
            if isinstance(closure, ResourceClosure):
                self._process_resource_closure(closure)
            else:
                self._process_closure(closure)
            del closure

    def create_resource(self, function, args=None, kwargs=None):
        """Asynchronously creates a per-worker resource represented by a `RemoteValue`.

    Args:
      function: the resource function to be run remotely. It should be a
        `tf.function`, a concrete function or a Python function.
      args: positional arguments to be passed to the function.
      kwargs: keyword arguments to be passed to the function.

    Returns:
      one or several RemoteValue objects depending on the function return
      values.
    """
        closure = ResourceClosure(function, self._cluster.resource_cancellation_mgr, args=args, kwargs=kwargs)
        return self._register_and_schedule_resource_closure(closure)

    def create_variable_resource(self, function, args=None, kwargs=None):
        """Create a per-worker variable."""
        closure = PerWorkerVariableClosure(function, self._cluster.resource_cancellation_mgr, args=args, kwargs=kwargs)
        return self._register_and_schedule_resource_closure(closure)

    def _register_and_schedule_resource_closure(self, closure):
        """Build remote value for, register for reconstruction, and schedule."""
        resource_remote_value = closure.build_output_remote_value()
        with self._resource_tracking_lock:
            self._register_resource(resource_remote_value)
            if self._is_dead_with_error:
                resource_remote_value._set_aborted(ClosureAbortedError(self._is_dead_with_error))
            else:
                self._schedule_resource(closure)
        return resource_remote_value

    def _register_resource(self, resource_remote_value):
        if not isinstance(resource_remote_value, RemoteValue):
            raise ValueError('Resource being registered is not of type `tf.distribute.experimental.coordinator.RemoteValue`.')
        self._resource_remote_value_refs.append(weakref.ref(resource_remote_value))