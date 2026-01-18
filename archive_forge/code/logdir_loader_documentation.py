import collections
import os
from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.util import tb_logging
Wraps `DirectoryLoader` generator to swallow
        `DirectoryDeletedError`.