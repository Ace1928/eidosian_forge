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
class RecordingVersionedFilesDecorator:
    """A minimal versioned files that records calls made on it.

    Only enough methods have been added to support tests using it to date.

    :ivar calls: A list of the calls made; can be reset at any time by
        assigning [] to it.
    """

    def __init__(self, backing_vf):
        """Create a RecordingVersionedFilesDecorator decorating backing_vf.

        :param backing_vf: The versioned file to answer all methods.
        """
        self._backing_vf = backing_vf
        self.calls = []

    def add_lines(self, key, parents, lines, parent_texts=None, left_matching_blocks=None, nostore_sha=None, random_id=False, check_content=True):
        self.calls.append(('add_lines', key, parents, lines, parent_texts, left_matching_blocks, nostore_sha, random_id, check_content))
        return self._backing_vf.add_lines(key, parents, lines, parent_texts, left_matching_blocks, nostore_sha, random_id, check_content)

    def add_content(self, factory, parent_texts=None, left_matching_blocks=None, nostore_sha=None, random_id=False, check_content=True):
        self.calls.append(('add_content', factory, parent_texts, left_matching_blocks, nostore_sha, random_id, check_content))
        return self._backing_vf.add_content(factory, parent_texts, left_matching_blocks, nostore_sha, random_id, check_content)

    def check(self):
        self._backing_vf.check()

    def get_parent_map(self, keys):
        self.calls.append(('get_parent_map', copy(keys)))
        return self._backing_vf.get_parent_map(keys)

    def get_record_stream(self, keys, sort_order, include_delta_closure):
        self.calls.append(('get_record_stream', list(keys), sort_order, include_delta_closure))
        return self._backing_vf.get_record_stream(keys, sort_order, include_delta_closure)

    def get_sha1s(self, keys):
        self.calls.append(('get_sha1s', copy(keys)))
        return self._backing_vf.get_sha1s(keys)

    def iter_lines_added_or_present_in_keys(self, keys, pb=None):
        self.calls.append(('iter_lines_added_or_present_in_keys', copy(keys)))
        return self._backing_vf.iter_lines_added_or_present_in_keys(keys, pb=pb)

    def keys(self):
        self.calls.append(('keys',))
        return self._backing_vf.keys()