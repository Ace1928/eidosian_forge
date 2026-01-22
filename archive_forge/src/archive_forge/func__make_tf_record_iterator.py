import contextlib
from tensorboard import data_compat
from tensorboard import dataclass_compat
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.util import platform_util
from tensorboard.util import tb_logging
def _make_tf_record_iterator(file_path):
    """Returns an iterator over TF records for the given tfrecord file."""
    if tf.__version__ == 'stub':
        logger.debug('Opening a stub record reader pointing at %s', file_path)
        return _PyRecordReaderIterator(tf.pywrap_tensorflow.PyRecordReader_New, file_path)
    try:
        from tensorflow.python import pywrap_tensorflow
        py_record_reader_new = pywrap_tensorflow.PyRecordReader_New
    except (ImportError, AttributeError):
        py_record_reader_new = None
    if py_record_reader_new:
        logger.debug('Opening a PyRecordReader pointing at %s', file_path)
        return _PyRecordReaderIterator(py_record_reader_new, file_path)
    else:
        logger.debug('Opening a tf_record_iterator pointing at %s', file_path)
        with _silence_deprecation_warnings():
            return tf.compat.v1.io.tf_record_iterator(file_path)