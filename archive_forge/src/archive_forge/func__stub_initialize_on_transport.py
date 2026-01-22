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
def _stub_initialize_on_transport(self, transport, file_mode):
    """Workaround: create control files for a remote working tree.

        This ensures that it can later be updated and dealt with locally,
        since BzrDirFormat6 and BzrDirFormat5 cannot represent dirs with
        no working tree.  (See bug #43064).
        """
    sio = BytesIO()
    inv = inventory.Inventory()
    xml5.serializer_v5.write_inventory(inv, sio, working=True)
    sio.seek(0)
    transport.put_file('inventory', sio, file_mode)
    transport.put_bytes('pending-merges', b'', file_mode)