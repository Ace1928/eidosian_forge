import time
import zlib
from typing import Type
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import errors, osutils, trace
from ..lru_cache import LRUSizeCache
from .btree_index import BTreeBuilder
from .versionedfile import (AbsentContentFactory, ChunkedContentFactory,
from ._groupcompress_py import (LinesDeltaIndex, apply_delta,
def _parse_bytes(self, data, pos):
    """Read the various lengths from the header.

        This also populates the various 'compressed' buffers.

        :return: The position in bytes just after the last newline
        """
    pos2 = data.index(b'\n', pos, pos + 14)
    self._z_content_length = int(data[pos:pos2])
    pos = pos2 + 1
    pos2 = data.index(b'\n', pos, pos + 14)
    self._content_length = int(data[pos:pos2])
    pos = pos2 + 1
    if len(data) != pos + self._z_content_length:
        raise AssertionError('Invalid bytes: (%d) != %d + %d' % (len(data), pos, self._z_content_length))
    self._z_content_chunks = (data[pos:],)