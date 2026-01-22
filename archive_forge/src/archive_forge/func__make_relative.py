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
@staticmethod
def _make_relative(base):
    """
        In order to ensure that a filename always honors the
        indicated output_dir, make sure it's relative.
        Ref python/cpython#37775.
        """
    no_drive = os.path.splitdrive(base)[1]
    return no_drive[os.path.isabs(no_drive):]