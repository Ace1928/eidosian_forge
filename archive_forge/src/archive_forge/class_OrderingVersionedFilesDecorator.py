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
class OrderingVersionedFilesDecorator(RecordingVersionedFilesDecorator):
    """A VF that records calls, and returns keys in specific order.

    :ivar calls: A list of the calls made; can be reset at any time by
        assigning [] to it.
    """

    def __init__(self, backing_vf, key_priority):
        """Create a RecordingVersionedFilesDecorator decorating backing_vf.

        :param backing_vf: The versioned file to answer all methods.
        :param key_priority: A dictionary defining what order keys should be
            returned from an 'unordered' get_record_stream request.
            Keys with lower priority are returned first, keys not present in
            the map get an implicit priority of 0, and are returned in
            lexicographical order.
        """
        RecordingVersionedFilesDecorator.__init__(self, backing_vf)
        self._key_priority = key_priority

    def get_record_stream(self, keys, sort_order, include_delta_closure):
        self.calls.append(('get_record_stream', list(keys), sort_order, include_delta_closure))
        if sort_order == 'unordered':

            def sort_key(key):
                return (self._key_priority.get(key, 0), key)
            for key in sorted(keys, key=sort_key):
                yield from self._backing_vf.get_record_stream([key], 'unordered', include_delta_closure)
        else:
            yield from self._backing_vf.get_record_stream(keys, sort_order, include_delta_closure)