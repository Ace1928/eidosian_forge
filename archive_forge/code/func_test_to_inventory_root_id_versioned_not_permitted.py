from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_to_inventory_root_id_versioned_not_permitted(self):
    root_entry = inventory.make_entry('directory', '', None, b'TREE_ROOT')
    root_entry.revision = b'some-version'
    delta = [(None, '', b'TREE_ROOT', root_entry)]
    serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=False, tree_references=True)
    self.assertRaises(InventoryDeltaError, serializer.delta_to_lines, b'old-version', b'new-version', delta)