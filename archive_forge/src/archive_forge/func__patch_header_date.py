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
def _patch_header_date(tree, path):
    """Returns a timestamp suitable for use in a patch header."""
    try:
        mtime = tree.get_file_mtime(path)
    except FileTimestampUnavailable:
        mtime = 0
    return timestamp.format_patch_date(mtime)