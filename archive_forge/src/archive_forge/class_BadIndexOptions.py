import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
class BadIndexOptions(errors.BzrError):
    _fmt = 'Could not parse options for index %(value)s.'

    def __init__(self, value):
        errors.BzrError.__init__(self)
        self.value = value