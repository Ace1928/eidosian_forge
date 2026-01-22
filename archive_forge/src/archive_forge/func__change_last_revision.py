import errno
from .. import errors, osutils
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from ..lockable_files import LockableFiles
from ..lockdir import LockDir
from ..mutabletree import MutableTree
from ..transport.local import LocalTransport
from . import bzrdir, hashcache, inventory
from . import transform as bzr_transform
from .workingtree import InventoryWorkingTree, WorkingTreeFormatMetaDir
def _change_last_revision(self, revision_id):
    """See WorkingTree._change_last_revision."""
    if revision_id is None or revision_id == _mod_revision.NULL_REVISION:
        try:
            self._transport.delete('last-revision')
        except _mod_transport.NoSuchFile:
            pass
        return False
    else:
        self._transport.put_bytes('last-revision', revision_id, mode=self.controldir._get_file_mode())
        return True