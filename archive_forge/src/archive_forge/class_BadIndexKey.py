import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
class BadIndexKey(errors.BzrError):
    _fmt = "The key '%(key)s' is not a valid key."

    def __init__(self, key):
        errors.BzrError.__init__(self)
        self.key = key