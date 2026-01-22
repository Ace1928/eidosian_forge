import contextlib
import errno
import os
import time
from stat import S_IEXEC, S_ISREG
from typing import Callable
from . import config as _mod_config
from . import controldir, errors, lazy_import, lock, osutils, registry, trace
from breezy import (
from breezy.i18n import gettext
from .errors import BzrError, DuplicateKey, InternalBzrError
from .filters import ContentFilterContext, filtered_output_bytes
from .mutabletree import MutableTree
from .osutils import delete_any, file_kind, pathjoin, sha_file, splitpath
from .progress import ProgressPhase
from .transport import FileExists, NoSuchFile
from .tree import InterTree, find_previous_path
class ImmortalLimbo(BzrError):
    _fmt = 'Unable to delete transform temporary directory %(limbo_dir)s.\n    Please examine %(limbo_dir)s to see if it contains any files you wish to\n    keep, and delete it when you are done.'

    def __init__(self, limbo_dir):
        BzrError.__init__(self)
        self.limbo_dir = limbo_dir