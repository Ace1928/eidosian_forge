import errno
import os
import re
from ..lazy_import import lazy_import
from breezy import (
from .. import transport as _mod_transport
from ..conflicts import Conflict as BaseConflict
from ..conflicts import ConflictList as BaseConflictList
from . import rio
class ParentLoop(HandledPathConflict):
    """An attempt to create an infinitely-looping directory structure.
    This is rare, but can be produced like so:

    tree A:
      mv foo bar
    tree B:
      mv bar foo
    merge A and B
    """
    typestring = 'parent loop'
    format = 'Conflict moving %(path)s into %(conflict_path)s. %(action)s.'

    def action_take_this(self, tree):
        pass

    def action_take_other(self, tree):
        with tree.transform() as tt:
            p_tid = tt.trans_id_file_id(self.file_id)
            parent_tid = tt.get_tree_parent(p_tid)
            cp_tid = tt.trans_id_file_id(self.conflict_file_id)
            cparent_tid = tt.get_tree_parent(cp_tid)
            tt.adjust_path(osutils.basename(self.path), cparent_tid, cp_tid)
            tt.adjust_path(osutils.basename(self.conflict_path), parent_tid, p_tid)
            tt.apply()