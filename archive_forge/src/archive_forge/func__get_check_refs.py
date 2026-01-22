from io import BytesIO
from ... import conflicts as _mod_conflicts
from ... import errors, lock, osutils
from ... import revision as _mod_revision
from ... import transport as _mod_transport
from ...bzr import conflicts as _mod_bzr_conflicts
from ...bzr import inventory
from ...bzr import transform as bzr_transform
from ...bzr import xml5
from ...bzr.workingtree_3 import PreDirStateWorkingTree
from ...mutabletree import MutableTree
from ...transport.local import LocalTransport
from ...workingtree import WorkingTreeFormat
def _get_check_refs(self):
    """Return the references needed to perform a check of this tree."""
    return [('trees', self.last_revision())]