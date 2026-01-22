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
def _write_file(self, relpath, tree, prefix, force_temp=False, allow_write=False):
    if not force_temp and isinstance(tree, WorkingTree):
        full_path = tree.abspath(relpath)
        if self._is_safepath(full_path):
            return full_path
    full_path = self._safe_filename(prefix, relpath)
    if not force_temp and self._try_symlink_root(tree, prefix):
        return full_path
    parent_dir = osutils.dirname(full_path)
    try:
        os.makedirs(parent_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    with tree.get_file(relpath) as source, open(full_path, 'wb') as target:
        osutils.pumpfile(source, target)
    try:
        mtime = tree.get_file_mtime(relpath)
    except FileTimestampUnavailable:
        pass
    else:
        os.utime(full_path, (mtime, mtime))
    if not allow_write:
        osutils.make_readonly(full_path)
    return full_path