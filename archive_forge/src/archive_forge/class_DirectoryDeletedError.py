import bisect
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.util import io_util
from tensorboard.util import tb_logging
class DirectoryDeletedError(Exception):
    """Thrown by Load() when the directory is *permanently* gone.

    We distinguish this from temporary errors so that other code can
    decide to drop all of our data only when a directory has been
    intentionally deleted, as opposed to due to transient filesystem
    errors.
    """
    pass