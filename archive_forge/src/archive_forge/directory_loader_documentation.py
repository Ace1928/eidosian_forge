from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.util import tb_logging
If max_timestamp is inactive, returns True and marks the path as
        such.