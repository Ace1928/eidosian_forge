import errno
import os
import re
from ..lazy_import import lazy_import
from breezy import (
from .. import transport as _mod_transport
from ..conflicts import Conflict as BaseConflict
from ..conflicts import ConflictList as BaseConflictList
from . import rio
def as_stanza(self):
    s = HandledConflict.as_stanza(self)
    s.add('conflict_path', self.conflict_path)
    if self.conflict_file_id is not None:
        s.add('conflict_file_id', self.conflict_file_id.decode('utf8'))
    return s