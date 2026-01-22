import itertools
import os
import struct
from copy import copy
from io import BytesIO
from typing import Any, Tuple
from zlib import adler32
from ..lazy_import import lazy_import
import fastbencode as bencode
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import graph as _mod_graph
from .. import osutils
from .. import transport as _mod_transport
from ..registry import Registry
from ..textmerge import TextMerge
from . import index
def _compute_diff(self, key, parent_lines, lines):
    """Compute a single mp_diff, and store it in self._diffs"""
    if len(parent_lines) > 0:
        left_parent_blocks = self.vf._extract_blocks(key, parent_lines[0], lines)
    else:
        left_parent_blocks = None
    diff = multiparent.MultiParent.from_lines(lines, parent_lines, left_parent_blocks)
    self.diffs[key] = diff