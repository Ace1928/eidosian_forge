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
class UnavailableRepresentation(errors.InternalBzrError):
    _fmt = "The encoding '%(wanted)s' is not available for key %(key)s which is encoded as '%(native)s'."

    def __init__(self, key, wanted, native):
        errors.InternalBzrError.__init__(self)
        self.wanted = wanted
        self.native = native
        self.key = key