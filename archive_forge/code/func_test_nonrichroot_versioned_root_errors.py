from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_nonrichroot_versioned_root_errors(self):
    old_inv = Inventory(None)
    new_inv = Inventory(None)
    root = new_inv.make_entry('directory', '', None, b'TREE_ROOT')
    root.revision = b'a@e\xc3\xa5ample.com--2004'
    new_inv.add(root)
    delta = new_inv._make_delta(old_inv)
    serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=False, tree_references=True)
    err = self.assertRaises(InventoryDeltaError, serializer.delta_to_lines, NULL_REVISION, b'entry-version', delta)
    self.assertContainsRe(str(err), "^Version present for / in b?'TREE_ROOT'")