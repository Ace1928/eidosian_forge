from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_reference_revision(self):
    entry = inventory.make_entry('tree-reference', 'a tree', None)
    entry.reference_revision = b'foo@\xc3\xa5b-lah'
    self.assertEqual(b'tree\x00foo@\xc3\xa5b-lah', inventory_delta._reference_content(entry))