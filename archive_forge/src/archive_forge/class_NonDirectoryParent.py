import errno
import os
import re
from ..lazy_import import lazy_import
from breezy import (
from .. import transport as _mod_transport
from ..conflicts import Conflict as BaseConflict
from ..conflicts import ConflictList as BaseConflictList
from . import rio
class NonDirectoryParent(HandledConflict):
    """An attempt to add files to a directory that is not a directory or
    an attempt to change the kind of a directory with files.
    """
    typestring = 'non-directory parent'
    format = 'Conflict: %(path)s is not a directory, but has files in it.  %(action)s.'

    def action_take_this(self, tree):
        if self.path.endswith('.new'):
            conflict_path = self.path[:-len('.new')]
            tree.remove([self.path], force=True, keep_files=False)
            tree.add(conflict_path)
        else:
            raise NotImplementedError(self.action_take_this)

    def action_take_other(self, tree):
        if self.path.endswith('.new'):
            conflict_path = self.path[:-len('.new')]
            tree.remove([conflict_path], force=True, keep_files=False)
            tree.rename_one(self.path, conflict_path)
        else:
            raise NotImplementedError(self.action_take_other)