from __future__ import absolute_import, division, unicode_literals
from future.builtins import bytes, chr, dict, int, range, str
from future.utils import raise_with_traceback
import re
import sys
import collections
from collections import namedtuple
class Quoter(collections.defaultdict):
    """A mapping from bytes (in range(0,256)) to strings.

    String values are percent-encoded byte values, unless the key < 128, and
    in the "safe" set (either the specified safe set, or default set).
    """

    def __init__(self, safe):
        """safe: bytes object."""
        self.safe = _ALWAYS_SAFE.union(bytes(safe))

    def __repr__(self):
        return '<Quoter %r>' % dict(self)

    def __missing__(self, b):
        res = chr(b) if b in self.safe else '%{0:02X}'.format(b)
        self[b] = res
        return res