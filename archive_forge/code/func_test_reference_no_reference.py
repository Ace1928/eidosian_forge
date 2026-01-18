from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_reference_no_reference(self):
    entry = inventory.make_entry('tree-reference', 'a tree', None)
    self.assertRaises(InventoryDeltaError, inventory_delta._reference_content, entry)