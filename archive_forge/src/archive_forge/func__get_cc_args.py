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
def _get_cc_args(self, pp_opts, debug, before):
    cc_args = pp_opts + ['-c']
    if debug:
        cc_args[:0] = ['-g']
    if before:
        cc_args[:0] = before
    return cc_args