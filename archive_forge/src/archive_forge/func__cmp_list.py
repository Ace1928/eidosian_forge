import errno
import os
import re
from ..lazy_import import lazy_import
from breezy import (
from .. import transport as _mod_transport
from ..conflicts import Conflict as BaseConflict
from ..conflicts import ConflictList as BaseConflictList
from . import rio
def _cmp_list(self):
    return HandledConflict._cmp_list(self) + [self.conflict_path, self.conflict_file_id]