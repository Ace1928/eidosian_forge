import contextlib
import difflib
import os
import re
import sys
from typing import List, Optional, Type, Union
from .lazy_import import lazy_import
import errno
import patiencediff
import subprocess
from breezy import (
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext
from . import errors, osutils
from . import transport as _mod_transport
from .registry import Registry
from .trace import mutter, note, warning
from .tree import FileTimestampUnavailable, Tree
def _try_symlink_root(self, tree, prefix):
    if getattr(tree, 'abspath', None) is None or not osutils.supports_symlinks(self._root):
        return False
    try:
        os.symlink(tree.abspath(''), osutils.pathjoin(self._root, prefix))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return True