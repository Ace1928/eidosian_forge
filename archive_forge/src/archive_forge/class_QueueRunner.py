import threading
import weakref
from tensorflow.core.protobuf import queue_runner_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.queue_runner.QueueRunner', 'train.QueueRunner'])
class QueueRunner:
    """Holds a list of enqueue operations for a queue, each to be run in a thread.

  Queues are a convenient TensorFlow mechanism to compute tensors
  asynchronously using multiple threads. For example in the canonical 'Input
  Reader' setup one set of threads generates filenames in a queue; a second set
  of threads read records from the files, processes them, and enqueues tensors
  on a second queue; a third set of threads dequeues these input records to
  construct batches and runs them through training operations.

  There are several delicate issues when running multiple threads that way:
  closing the queues in sequence as the input is exhausted, correctly catching
  and reporting exceptions, etc.

  The `QueueRunner`, combined with the `Coordinator`, helps handle these issues.

  @compatibility(TF2)
  QueueRunners are not compatible with eager execution. Instead, please
  use [tf.data](https://www.tensorflow.org/guide/data) to get data into your
  model.
  @end_compatibility
  """

    @deprecation.deprecated(None, _DEPRECATION_INSTRUCTION)
    def __init__(self, queue=None, enqueue_ops=None, close_op=None, cancel_op=None, queue_closed_exception_types=None, queue_runner_def=None, import_scope=None):
        """Create a QueueRunner.

    On construction the `QueueRunner` adds an op to close the queue.  That op
    will be run if the enqueue ops raise exceptions.

    When you later call the `create_threads()` method, the `QueueRunner` will
    create one thread for each op in `enqueue_ops`.  Each thread will run its
    enqueue op in parallel with the other threads.  The enqueue ops do not have
    to all be the same op, but it is expected that they all enqueue tensors in
    `queue`.

    Args:
      queue: A `Queue`.
      enqueue_ops: List of enqueue ops to run in threads later.
      close_op: Op to close the queue. Pending enqueue ops are preserved.
      cancel_op: Op to close the queue and cancel pending enqueue ops.
      queue_closed_exception_types: Optional tuple of Exception types that
        indicate that the queue has been closed when raised during an enqueue
        operation.  Defaults to `(tf.errors.OutOfRangeError,)`.  Another common
        case includes `(tf.errors.OutOfRangeError, tf.errors.CancelledError)`,
        when some of the enqueue ops may dequeue from other Queues.
      queue_runner_def: Optional `QueueRunnerDef` protocol buffer. If specified,
        recreates the QueueRunner from its contents. `queue_runner_def` and the
        other arguments are mutually exclusive.
      import_scope: Optional `string`. Name scope to add. Only used when
        initializing from protocol buffer.

    Raises:
      ValueError: If both `queue_runner_def` and `queue` are both specified.
      ValueError: If `queue` or `enqueue_ops` are not provided when not
        restoring from `queue_runner_def`.
      RuntimeError: If eager execution is enabled.
    """
        if context.executing_eagerly():
            raise RuntimeError('QueueRunners are not supported when eager execution is enabled. Instead, please use tf.data to get data into your model.')
        if queue_runner_def:
            if queue or enqueue_ops:
                raise ValueError('queue_runner_def and queue are mutually exclusive.')
            self._init_from_proto(queue_runner_def, import_scope=import_scope)
        else:
            self._init_from_args(queue=queue, enqueue_ops=enqueue_ops, close_op=close_op, cancel_op=cancel_op, queue_closed_exception_types=queue_closed_exception_types)
        self._lock = threading.Lock()
        self._runs_per_session = weakref.WeakKeyDictionary()
        self._exceptions_raised = []

    def _init_from_args(self, queue=None, enqueue_ops=None, close_op=None, cancel_op=None, queue_closed_exception_types=None):
        """Create a QueueRunner from arguments.

    Args:
      queue: A `Queue`.
      enqueue_ops: List of enqueue ops to run in threads later.
      close_op: Op to close the queue. Pending enqueue ops are preserved.
      cancel_op: Op to close the queue and cancel pending enqueue ops.
      queue_closed_exception_types: Tuple of exception types, which indicate
        the queue has been safely closed.

    Raises:
      ValueError: If `queue` or `enqueue_ops` are not provided when not
        restoring from `queue_runner_def`.
      TypeError: If `queue_closed_exception_types` is provided, but is not
        a non-empty tuple of error types (subclasses of `tf.errors.OpError`).
    """
        if not queue or not enqueue_ops:
            raise ValueError('Must provide queue and enqueue_ops.')
        self._queue = queue
        self._enqueue_ops = enqueue_ops
        self._close_op = close_op
        self._cancel_op = cancel_op
        if queue_closed_exception_types is not None:
            if not isinstance(queue_closed_exception_types, tuple) or not queue_closed_exception_types or (not all((issubclass(t, errors.OpError) for t in queue_closed_exception_types))):
                raise TypeError('queue_closed_exception_types, when provided, must be a tuple of tf.error types, but saw: %s' % queue_closed_exception_types)
        self._queue_closed_exception_types = queue_closed_exception_types
        if self._close_op is None:
            self._close_op = self._queue.close()
        if self._cancel_op is None:
            self._cancel_op = self._queue.close(cancel_pending_enqueues=True)
        if not self._queue_closed_exception_types:
            self._queue_closed_exception_types = (errors.OutOfRangeError,)
        else:
            self._queue_closed_exception_types = tuple(self._queue_closed_exception_types)

    def _init_from_proto(self, queue_runner_def, import_scope=None):
        """Create a QueueRunner from `QueueRunnerDef`.

    Args:
      queue_runner_def: Optional `QueueRunnerDef` protocol buffer.
      import_scope: Optional `string`. Name scope to add.
    """
        assert isinstance(queue_runner_def, queue_runner_pb2.QueueRunnerDef)
        g = ops.get_default_graph()
        self._queue = g.as_graph_element(ops.prepend_name_scope(queue_runner_def.queue_name, import_scope))
        self._enqueue_ops = [g.as_graph_element(ops.prepend_name_scope(op, import_scope)) for op in queue_runner_def.enqueue_op_name]
        self._close_op = g.as_graph_element(ops.prepend_name_scope(queue_runner_def.close_op_name, import_scope))
        self._cancel_op = g.as_graph_element(ops.prepend_name_scope(queue_runner_def.cancel_op_name, import_scope))
        self._queue_closed_exception_types = tuple((errors.exception_type_from_error_code(code) for code in queue_runner_def.queue_closed_exception_types))
        if not self._queue_closed_exception_types:
            self._queue_closed_exception_types = (errors.OutOfRangeError,)

    @property
    def queue(self):
        return self._queue

    @property
    def enqueue_ops(self):
        return self._enqueue_ops

    @property
    def close_op(self):
        return self._close_op

    @property
    def cancel_op(self):
        return self._cancel_op

    @property
    def queue_closed_exception_types(self):
        return self._queue_closed_exception_types

    @property
    def exceptions_raised(self):
        """Exceptions raised but not handled by the `QueueRunner` threads.

    Exceptions raised in queue runner threads are handled in one of two ways
    depending on whether or not a `Coordinator` was passed to
    `create_threads()`:

    * With a `Coordinator`, exceptions are reported to the coordinator and
      forgotten by the `QueueRunner`.
    * Without a `Coordinator`, exceptions are captured by the `QueueRunner` and
      made available in this `exceptions_raised` property.

    Returns:
      A list of Python `Exception` objects.  The list is empty if no exception
      was captured.  (No exceptions are captured when using a Coordinator.)
    """
        return self._exceptions_raised

    @property
    def name(self):
        """The string name of the underlying Queue."""
        return self._queue.name

    def _run(self, sess, enqueue_op, coord=None):
        """Execute the enqueue op in a loop, close the queue in case of error.

    Args:
      sess: A Session.
      enqueue_op: The Operation to run.
      coord: Optional Coordinator object for reporting errors and checking
        for stop conditions.
    """
        decremented = False
        try:
            enqueue_callable = sess.make_callable(enqueue_op)
            while True:
                if coord and coord.should_stop():
                    break
                try:
                    enqueue_callable()
                except self._queue_closed_exception_types:
                    with self._lock:
                        self._runs_per_session[sess] -= 1
                        decremented = True
                        if self._runs_per_session[sess] == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception as e:
                                logging.vlog(1, 'Ignored exception: %s', str(e))
                        return
        except Exception as e:
            if coord:
                coord.request_stop(e)
            else:
                logging.error('Exception in QueueRunner: %s', str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            if not decremented:
                with self._lock:
                    self._runs_per_session[sess] -= 1

    def _close_on_stop(self, sess, cancel_op, coord):
        """Close the queue when the Coordinator requests stop.

    Args:
      sess: A Session.
      cancel_op: The Operation to run.
      coord: Coordinator.
    """
        coord.wait_for_stop()
        try:
            sess.run(cancel_op)
        except Exception as e:
            logging.vlog(1, 'Ignored exception: %s', str(e))

    def create_threads(self, sess, coord=None, daemon=False, start=False):
        """Create threads to run the enqueue ops for the given session.

    This method requires a session in which the graph was launched.  It creates
    a list of threads, optionally starting them.  There is one thread for each
    op passed in `enqueue_ops`.

    The `coord` argument is an optional coordinator that the threads will use
    to terminate together and report exceptions.  If a coordinator is given,
    this method starts an additional thread to close the queue when the
    coordinator requests a stop.

    If previously created threads for the given session are still running, no
    new threads will be created.

    Args:
      sess: A `Session`.
      coord: Optional `Coordinator` object for reporting errors and checking
        stop conditions.
      daemon: Boolean.  If `True` make the threads daemon threads.
      start: Boolean.  If `True` starts the threads.  If `False` the
        caller must call the `start()` method of the returned threads.

    Returns:
      A list of threads.
    """
        with self._lock:
            try:
                if self._runs_per_session[sess] > 0:
                    return []
            except KeyError:
                pass
            self._runs_per_session[sess] = len(self._enqueue_ops)
            self._exceptions_raised = []
        ret_threads = []
        for op in self._enqueue_ops:
            name = 'QueueRunnerThread-{}-{}'.format(self.name, op.name)
            ret_threads.append(threading.Thread(target=self._run, args=(sess, op, coord), name=name))
        if coord:
            name = 'QueueRunnerThread-{}-close_on_stop'.format(self.name)
            ret_threads.append(threading.Thread(target=self._close_on_stop, args=(sess, self._cancel_op, coord), name=name))
        for t in ret_threads:
            if coord:
                coord.register_thread(t)
            if daemon:
                t.daemon = True
            if start:
                t.start()
        return ret_threads

    def to_proto(self, export_scope=None):
        """Converts this `QueueRunner` to a `QueueRunnerDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Returns:
      A `QueueRunnerDef` protocol buffer, or `None` if the `Variable` is not in
      the specified name scope.
    """
        if export_scope is None or self.queue.name.startswith(export_scope):
            queue_runner_def = queue_runner_pb2.QueueRunnerDef()
            queue_runner_def.queue_name = ops.strip_name_scope(self.queue.name, export_scope)
            for enqueue_op in self.enqueue_ops:
                queue_runner_def.enqueue_op_name.append(ops.strip_name_scope(enqueue_op.name, export_scope))
            queue_runner_def.close_op_name = ops.strip_name_scope(self.close_op.name, export_scope)
            queue_runner_def.cancel_op_name = ops.strip_name_scope(self.cancel_op.name, export_scope)
            queue_runner_def.queue_closed_exception_types.extend([errors.error_code_from_exception_type(cls) for cls in self._queue_closed_exception_types])
            return queue_runner_def
        else:
            return None

    @staticmethod
    def from_proto(queue_runner_def, import_scope=None):
        """Returns a `QueueRunner` object created from `queue_runner_def`."""
        return QueueRunner(queue_runner_def=queue_runner_def, import_scope=import_scope)