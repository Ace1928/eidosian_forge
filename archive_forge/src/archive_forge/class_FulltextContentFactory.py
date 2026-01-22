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
class FulltextContentFactory(ContentFactory):
    """Static data content factory.

    This takes a fulltext when created and just returns that during
    get_bytes_as('fulltext').

    :ivar sha1: None, or the sha1 of the content fulltext.
    :ivar storage_kind: The native storage kind of this factory. Always
        'fulltext'.
    :ivar key: The key of this content. Each key is a tuple with a single
        string in it.
    :ivar parents: A tuple of parent keys for self.key. If the object has
        no parent information, None (as opposed to () for an empty list of
        parents).
     """

    def __init__(self, key, parents, sha1, text):
        """Create a ContentFactory."""
        self.sha1 = sha1
        self.size = len(text)
        self.storage_kind = 'fulltext'
        self.key = key
        self.parents = parents
        if not isinstance(text, bytes):
            raise TypeError(text)
        self._text = text

    def get_bytes_as(self, storage_kind):
        if storage_kind == self.storage_kind:
            return self._text
        elif storage_kind == 'chunked':
            return [self._text]
        elif storage_kind == 'lines':
            return osutils.split_lines(self._text)
        raise UnavailableRepresentation(self.key, storage_kind, self.storage_kind)

    def iter_bytes_as(self, storage_kind):
        if storage_kind == 'chunked':
            return iter([self._text])
        elif storage_kind == 'lines':
            return iter(osutils.split_lines(self._text))
        raise UnavailableRepresentation(self.key, storage_kind, self.storage_kind)