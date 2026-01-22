import os
import stat
from . import errors, lock
from . import revision as _mod_revision
from . import transport as _mod_transport
from .bzr.inventory import Inventory
from .bzr.inventorytree import MutableInventoryTree
from .osutils import sha_file
from .transport.memory import MemoryTransport
def _set_basis(self):
    try:
        self._basis_tree = self.branch.repository.revision_tree(self._branch_revision_id)
    except errors.NoSuchRevision:
        if self._allow_leftmost_as_ghost:
            self._basis_tree = self.branch.repository.revision_tree(_mod_revision.NULL_REVISION)
        else:
            raise