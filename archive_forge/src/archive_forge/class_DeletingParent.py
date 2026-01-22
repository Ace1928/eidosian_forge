import errno
import os
import re
from ..lazy_import import lazy_import
from breezy import (
from .. import transport as _mod_transport
from ..conflicts import Conflict as BaseConflict
from ..conflicts import ConflictList as BaseConflictList
from . import rio
class DeletingParent(HandledConflict):
    """An attempt to add files to a directory that is not present.
    Typically, the result of a merge where one OTHER deleted the directory and
    the THIS added a file to it.
    """
    typestring = 'deleting parent'
    format = "Conflict: can't delete %(path)s because it is not empty.  %(action)s."

    def action_take_this(self, tree):
        pass

    def action_take_other(self, tree):
        tree.remove([self.path], force=True, keep_files=False)