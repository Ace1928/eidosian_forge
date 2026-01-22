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
def _write_hashcache_if_dirty(self):
    """Write out the hashcache if it is dirty."""
    if self._hashcache.needs_write:
        try:
            self._hashcache.write()
        except OSError as e:
            if e.errno not in (errno.EPERM, errno.EACCES):
                raise
            trace.mutter('Could not write hashcache for %s\nError: %s', self._hashcache.cache_file_name(), osutils.safe_unicode(e.args[1]))