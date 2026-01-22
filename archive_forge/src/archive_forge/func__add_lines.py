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
def _add_lines(self, version_id, parents, lines, parent_texts, left_matching_blocks, nostore_sha, random_id, check_content):
    """Helper to do the class specific add_lines."""
    raise NotImplementedError(self.add_lines)