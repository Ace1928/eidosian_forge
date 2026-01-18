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
Open the tree itself.

        :param a_controldir: the dir for the tree.
        :param control_files: the control files for the tree.
        