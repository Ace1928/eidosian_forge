import contextlib
from tensorboard import data_compat
from tensorboard import dataclass_compat
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.util import platform_util
from tensorboard.util import tb_logging
def CheckForIncreasedFileSize(self):
    """Stats the file to get its updated size, returning True if it grew.

        If the stat call fails or reports a smaller size than was previously
        seen, then any previously cached size is left unchanged.

        Returns:
            boolean or None: True if the file size increased; False if it was
                the same or decreased; or None if neither case could be detected
                (either because the previous size had not been recorded yet, or
                because the stat call for the current size failed).
        """
    previous_size = self._file_size
    try:
        self._file_size = tf.io.gfile.stat(self._file_path).length
    except tf.errors.OpError as e:
        logger.error('Failed to stat %s: %s', self._file_path, e)
        return None
    logger.debug('Stat on %s got size %d, previous size %s', self._file_path, self._file_size, previous_size)
    if previous_size is None:
        return None
    if self._file_size > previous_size:
        return True
    if self._file_size < previous_size:
        logger.warning('File %s shrank from previous size %d to size %d', self._file_path, previous_size, self._file_size)
        self._file_size = previous_size
    return False