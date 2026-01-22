import bisect
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.util import io_util
from tensorboard.util import tb_logging
def _LoadInternal(self):
    """Internal implementation of Load().

        The only difference between this and Load() is that the latter will throw
        DirectoryDeletedError on I/O errors if it thinks that the directory has been
        permanently deleted.

        Yields:
          All values that have not been yielded yet.
        """
    if not self._loader:
        self._InitializeLoader()
    if not self._loader:
        return
    while True:
        for event in self._loader.Load():
            yield event
        next_path = self._GetNextPath()
        if not next_path:
            logger.info('No path found after %s', self._path)
            return
        for event in self._loader.Load():
            yield event
        logger.info('Directory watcher advancing from %s to %s', self._path, next_path)
        self._SetPath(next_path)