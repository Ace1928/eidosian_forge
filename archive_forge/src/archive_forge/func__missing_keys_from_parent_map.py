import re
from bisect import bisect_right
from io import BytesIO
from ..lazy_import import lazy_import
from breezy import (
from .. import debug, errors
from .. import transport as _mod_transport
from .static_tuple import StaticTuple
def _missing_keys_from_parent_map(self, keys):
    return set(keys) - set(self.get_parent_map(keys))