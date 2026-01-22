import errno
import os
import re
from ..lazy_import import lazy_import
from breezy import (
from .. import transport as _mod_transport
from ..conflicts import Conflict as BaseConflict
from ..conflicts import ConflictList as BaseConflictList
from . import rio
class MissingParent(HandledConflict):
    """An attempt to add files to a directory that is not present.
    Typically, the result of a merge where THIS deleted the directory and
    the OTHER added a file to it.
    See also: DeletingParent (same situation, THIS and OTHER reversed)
    """
    typestring = 'missing parent'
    format = 'Conflict adding files to %(path)s.  %(action)s.'

    def action_take_this(self, tree):
        tree.remove([self.path], force=True, keep_files=False)

    def action_take_other(self, tree):
        pass