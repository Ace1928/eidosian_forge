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
def _iter_all_prefixes(self):
    if isinstance(self._mapper, ConstantMapper):
        paths = [self._mapper.map(())]
        prefixes = [()]
    else:
        relpaths = set()
        for quoted_relpath in self._transport.iter_files_recursive():
            path, ext = os.path.splitext(quoted_relpath)
            relpaths.add(path)
        paths = list(relpaths)
        prefixes = [self._mapper.unmap(path) for path in paths]
    return zip(paths, prefixes)