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
class AdapterFactory(ContentFactory):
    """A content factory to adapt between key prefix's."""

    def __init__(self, key, parents, adapted):
        """Create an adapter factory instance."""
        self.key = key
        self.parents = parents
        self._adapted = adapted

    def __getattr__(self, attr):
        """Return a member from the adapted object."""
        if attr in ('key', 'parents'):
            return self.__dict__[attr]
        else:
            return getattr(self._adapted, attr)