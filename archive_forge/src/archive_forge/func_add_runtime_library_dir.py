import sys
import os
import re
import warnings
from .errors import (
from .spawn import spawn
from .file_util import move_file
from .dir_util import mkpath
from ._modified import newer_group
from .util import split_quoted, execute
from ._log import log
def add_runtime_library_dir(self, dir):
    """Add 'dir' to the list of directories that will be searched for
        shared libraries at runtime.
        """
    self.runtime_library_dirs.append(dir)