import bisect
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.util import io_util
from tensorboard.util import tb_logging
def _GetNextPath(self):
    """Gets the next path to load from.

        This function also does the checking for out-of-order writes as it iterates
        through the paths.

        Returns:
          The next path to load events from, or None if there are no more paths.
        """
    paths = sorted((path for path in io_wrapper.ListDirectoryAbsolute(self._directory) if self._path_filter(path)))
    if not paths:
        return None
    if self._path is None:
        return paths[0]
    if not io_util.IsCloudPath(paths[0]) and (not self._ooo_writes_detected):
        current_path_index = bisect.bisect_left(paths, self._path)
        ooo_check_start = max(0, current_path_index - self._OOO_WRITE_CHECK_COUNT)
        for path in paths[ooo_check_start:current_path_index]:
            if self._HasOOOWrite(path):
                self._ooo_writes_detected = True
                break
    next_paths = list((path for path in paths if self._path is None or path > self._path))
    if next_paths:
        return min(next_paths)
    else:
        return None