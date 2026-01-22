from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops.gen_io_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['ReaderBase'])
class ReaderBase:
    """Base class for different Reader types, that produce a record every step.

  Conceptually, Readers convert string 'work units' into records (key,
  value pairs).  Typically the 'work units' are filenames and the
  records are extracted from the contents of those files.  We want a
  single record produced per step, but a work unit can correspond to
  many records.

  Therefore we introduce some decoupling using a queue.  The queue
  contains the work units and the Reader dequeues from the queue when
  it is asked to produce a record (via Read()) but it has finished the
  last work unit.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  """

    def __init__(self, reader_ref, supports_serialize=False):
        """Creates a new ReaderBase.

    Args:
      reader_ref: The operation that implements the reader.
      supports_serialize: True if the reader implementation can
        serialize its state.

    Raises:
      RuntimeError: If eager execution is enabled.
    """
        if context.executing_eagerly():
            raise RuntimeError('Readers are not supported when eager execution is enabled. Instead, please use tf.data to get data into your model.')
        self._reader_ref = reader_ref
        self._supports_serialize = supports_serialize

    @property
    def reader_ref(self):
        """Op that implements the reader."""
        return self._reader_ref

    def read(self, queue, name=None):
        """Returns the next record (key, value) pair produced by a reader.

    Will dequeue a work unit from queue if necessary (e.g. when the
    Reader needs to start reading from a new file since it has
    finished with the previous file).

    Args:
      queue: A Queue or a mutable string Tensor representing a handle
        to a Queue, with string work items.
      name: A name for the operation (optional).

    Returns:
      A tuple of Tensors (key, value).
      key: A string scalar Tensor.
      value: A string scalar Tensor.
    """
        if isinstance(queue, tensor_lib.Tensor):
            queue_ref = queue
        else:
            queue_ref = queue.queue_ref
        if self._reader_ref.dtype == dtypes.resource:
            return gen_io_ops.reader_read_v2(self._reader_ref, queue_ref, name=name)
        else:
            old_queue_op = gen_data_flow_ops.fake_queue(queue_ref)
            return gen_io_ops.reader_read(self._reader_ref, old_queue_op, name=name)

    def read_up_to(self, queue, num_records, name=None):
        """Returns up to num_records (key, value) pairs produced by a reader.

    Will dequeue a work unit from queue if necessary (e.g., when the
    Reader needs to start reading from a new file since it has
    finished with the previous file).
    It may return less than num_records even before the last batch.

    Args:
      queue: A Queue or a mutable string Tensor representing a handle
        to a Queue, with string work items.
      num_records: Number of records to read.
      name: A name for the operation (optional).

    Returns:
      A tuple of Tensors (keys, values).
      keys: A 1-D string Tensor.
      values: A 1-D string Tensor.
    """
        if isinstance(queue, tensor_lib.Tensor):
            queue_ref = queue
        else:
            queue_ref = queue.queue_ref
        if self._reader_ref.dtype == dtypes.resource:
            return gen_io_ops.reader_read_up_to_v2(self._reader_ref, queue_ref, num_records, name=name)
        else:
            old_queue_op = gen_data_flow_ops.fake_queue(queue_ref)
            return gen_io_ops.reader_read_up_to(self._reader_ref, old_queue_op, num_records, name=name)

    def num_records_produced(self, name=None):
        """Returns the number of records this reader has produced.

    This is the same as the number of Read executions that have
    succeeded.

    Args:
      name: A name for the operation (optional).

    Returns:
      An int64 Tensor.

    """
        if self._reader_ref.dtype == dtypes.resource:
            return gen_io_ops.reader_num_records_produced_v2(self._reader_ref, name=name)
        else:
            return gen_io_ops.reader_num_records_produced(self._reader_ref, name=name)

    def num_work_units_completed(self, name=None):
        """Returns the number of work units this reader has finished processing.

    Args:
      name: A name for the operation (optional).

    Returns:
      An int64 Tensor.
    """
        if self._reader_ref.dtype == dtypes.resource:
            return gen_io_ops.reader_num_work_units_completed_v2(self._reader_ref, name=name)
        else:
            return gen_io_ops.reader_num_work_units_completed(self._reader_ref, name=name)

    def serialize_state(self, name=None):
        """Produce a string tensor that encodes the state of a reader.

    Not all Readers support being serialized, so this can produce an
    Unimplemented error.

    Args:
      name: A name for the operation (optional).

    Returns:
      A string Tensor.
    """
        if self._reader_ref.dtype == dtypes.resource:
            return gen_io_ops.reader_serialize_state_v2(self._reader_ref, name=name)
        else:
            return gen_io_ops.reader_serialize_state(self._reader_ref, name=name)

    def restore_state(self, state, name=None):
        """Restore a reader to a previously saved state.

    Not all Readers support being restored, so this can produce an
    Unimplemented error.

    Args:
      state: A string Tensor.
        Result of a SerializeState of a Reader with matching type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.
    """
        if self._reader_ref.dtype == dtypes.resource:
            return gen_io_ops.reader_restore_state_v2(self._reader_ref, state, name=name)
        else:
            return gen_io_ops.reader_restore_state(self._reader_ref, state, name=name)

    @property
    def supports_serialize(self):
        """Whether the Reader implementation can serialize its state."""
        return self._supports_serialize

    def reset(self, name=None):
        """Restore a reader to its initial clean state.

    Args:
      name: A name for the operation (optional).

    Returns:
      The created Operation.
    """
        if self._reader_ref.dtype == dtypes.resource:
            return gen_io_ops.reader_reset_v2(self._reader_ref, name=name)
        else:
            return gen_io_ops.reader_reset(self._reader_ref, name=name)