import os.path
import logging
import re
from .patterns import match_any_paths
from wandb_watchdog.utils import has_attribute
from wandb_watchdog.utils import unicode_paths
class LoggingFileSystemEventHandler(LoggingEventHandler):
    """
    For backwards-compatibility. Please use :class:`LoggingEventHandler`
    instead.
    """