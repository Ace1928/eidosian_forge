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
def _transitive_fallbacks(self):
    """Return the whole stack of fallback versionedfiles.

        This VersionedFiles may have a list of fallbacks, but it doesn't
        necessarily know about the whole stack going down, and it can't know
        at open time because they may change after the objects are opened.
        """
    all_fallbacks = []
    for a_vfs in self._immediate_fallback_vfs:
        all_fallbacks.append(a_vfs)
        all_fallbacks.extend(a_vfs._transitive_fallbacks())
    return all_fallbacks