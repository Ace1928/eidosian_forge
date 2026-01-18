import errno
import os
import re
from ..lazy_import import lazy_import
from breezy import (
from .. import transport as _mod_transport
from ..conflicts import Conflict as BaseConflict
from ..conflicts import ConflictList as BaseConflictList
from . import rio
def to_stanzas(self):
    """Generator of stanzas"""
    for conflict in self:
        yield conflict.as_stanza()