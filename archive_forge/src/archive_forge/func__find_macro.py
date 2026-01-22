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
def _find_macro(self, name):
    i = 0
    for defn in self.macros:
        if defn[0] == name:
            return i
        i += 1
    return None