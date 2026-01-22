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
class PrefixMapper(URLEscapeMapper):
    """A key mapper that extracts the first component of a key.

    This mapper is for use with a transport based backend.
    """

    def _map(self, key):
        """See KeyMapper.map()."""
        return key[0].decode('utf-8')

    def _unmap(self, partition_id):
        """See KeyMapper.unmap()."""
        return (partition_id.encode('utf-8'),)