import contextlib
from tensorboard import data_compat
from tensorboard import dataclass_compat
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
from tensorboard.util import platform_util
from tensorboard.util import tb_logging
class RawEventFileLoader:
    """An iterator that yields Event protos as serialized bytestrings."""

    def __init__(self, file_path, detect_file_replacement=False):
        """Constructs a RawEventFileLoader for the given file path.

        Args:
          file_path: the event file path to read from
          detect_file_replacement: if True, when Load() is called, the loader
              will make a stat() call to check the size of the file. If it sees
              that the file has grown, it will reopen the file entirely (while
              preserving the current offset) before attempting to read from it.
              Otherwise, Load() will simply poll at EOF for new data.
        """
        if file_path is None:
            raise ValueError('A file path is required')
        self._file_path = platform_util.readahead_file_path(file_path)
        self._detect_file_replacement = detect_file_replacement
        self._file_size = None
        self._iterator = _make_tf_record_iterator(self._file_path)
        if self._detect_file_replacement and (not hasattr(self._iterator, 'reopen')):
            logger.warning('File replacement detection requested, but not enabled because TF record iterator impl does not support reopening. This functionality requires TensorFlow 2.9+')
            self._detect_file_replacement = False

    def Load(self):
        """Loads all new events from disk as raw serialized proto bytestrings.

        Calling Load multiple times in a row will not 'drop' events as long as the
        return value is not iterated over.

        Yields:
          All event proto bytestrings in the file that have not been yielded yet.
        """
        logger.debug('Loading events from %s', self._file_path)
        if self._detect_file_replacement:
            has_increased = self.CheckForIncreasedFileSize()
            if has_increased is not None:
                if has_increased:
                    logger.debug('Reopening %s since file size has changed', self._file_path)
                    self._iterator.close()
                    self._iterator.reopen()
                else:
                    logger.debug('Skipping attempt to poll %s since file size has not changed (still %d)', self._file_path, self._file_size)
                    return
        while True:
            try:
                yield next(self._iterator)
            except StopIteration:
                logger.debug('End of file in %s', self._file_path)
                break
            except tf.errors.DataLossError as e:
                logger.debug('Truncated record in %s (%s)', self._file_path, e)
                break
        logger.debug('No more events in %s', self._file_path)

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