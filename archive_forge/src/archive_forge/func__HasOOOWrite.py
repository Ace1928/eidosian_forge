import bisect
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.util import io_util
from tensorboard.util import tb_logging
def _HasOOOWrite(self, path):
    """Returns whether the path has had an out-of-order write."""
    size = tf.io.gfile.stat(path).length
    old_size = self._finalized_sizes.get(path, None)
    if size != old_size:
        if old_size is None:
            logger.error("File %s created after file %s even though it's lexicographically earlier", path, self._path)
        else:
            logger.error('File %s updated even though the current file is %s', path, self._path)
        return True
    else:
        return False