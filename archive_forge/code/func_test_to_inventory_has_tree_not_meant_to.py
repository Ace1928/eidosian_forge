from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_to_inventory_has_tree_not_meant_to(self):
    make_entry = inventory.make_entry
    tree_ref = make_entry('tree-reference', 'foo', b'changed-in', b'ref-id')
    tree_ref.reference_revision = b'ref-revision'
    delta = [(None, '', b'an-id', make_entry('directory', '', b'changed-in', b'an-id')), (None, 'foo', b'ref-id', tree_ref)]
    serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=True, tree_references=True)
    self.assertRaises(InventoryDeltaError, serializer.delta_to_lines, b'old-version', b'new-version', delta)