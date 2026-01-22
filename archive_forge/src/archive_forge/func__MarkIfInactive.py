from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.util import tb_logging
def _MarkIfInactive(self, path, max_timestamp):
    """If max_timestamp is inactive, returns True and marks the path as
        such."""
    logger.debug('Checking active status of %s at %s', path, max_timestamp)
    if max_timestamp is not None and (not self._active_filter(max_timestamp)):
        self._max_timestamps[path] = _INACTIVE
        del self._loaders[path]
        return True
    return False