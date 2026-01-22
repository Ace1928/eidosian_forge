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
def _process_one_record(self, key, this_chunks):
    parent_keys = None
    if key in self.parent_map:
        parent_keys = self.parent_map.pop(key)
        if parent_keys is None:
            parent_keys = ()
        parent_lines = []
        for p in parent_keys:
            if p in self.ghost_parents:
                continue
            refcount = self.refcounts[p]
            if refcount == 1:
                self.refcounts.pop(p)
                parent_chunks = self.chunks.pop(p)
            else:
                self.refcounts[p] = refcount - 1
                parent_chunks = self.chunks[p]
            p_lines = osutils.chunks_to_lines(parent_chunks)
            parent_lines.append(p_lines)
            del p_lines
        lines = osutils.chunks_to_lines(this_chunks)
        this_chunks = lines
        self._compute_diff(key, parent_lines, lines)
        del lines
    if key in self.refcounts:
        self.chunks[key] = this_chunks