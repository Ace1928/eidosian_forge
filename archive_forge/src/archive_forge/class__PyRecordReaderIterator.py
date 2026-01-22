import contextlib
from tensorboard import data_compat
from tensorboard import dataclass_compat
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.util import platform_util
from tensorboard.util import tb_logging
class _PyRecordReaderIterator:
    """Python iterator for TF Records based on PyRecordReader."""

    def __init__(self, py_record_reader_new, file_path):
        """Constructs a _PyRecordReaderIterator for the given file path.

        Args:
          py_record_reader_new: pywrap_tensorflow.PyRecordReader_New
          file_path: file path of the tfrecord file to read
        """
        with tf.compat.v1.errors.raise_exception_on_not_ok_status() as status:
            self._reader = py_record_reader_new(tf.compat.as_bytes(file_path), 0, tf.compat.as_bytes(''), status)
        if not self._reader:
            raise IOError('Failed to open a record reader pointing to %s' % file_path)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self._reader.GetNext()
        except tf.errors.OutOfRangeError as e:
            raise StopIteration
        return self._reader.record()
    next = __next__