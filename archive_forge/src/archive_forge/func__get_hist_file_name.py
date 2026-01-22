import atexit
import datetime
import re
import sqlite3
import threading
from pathlib import Path
from decorator import decorator
from traitlets import (
from traitlets.config.configurable import LoggingConfigurable
from IPython.paths import locate_profile
from IPython.utils.decorators import undoc
def _get_hist_file_name(self, profile=None):
    """Get default history file name based on the Shell's profile.

        The profile parameter is ignored, but must exist for compatibility with
        the parent class."""
    profile_dir = self.shell.profile_dir.location
    return Path(profile_dir) / 'history.sqlite'