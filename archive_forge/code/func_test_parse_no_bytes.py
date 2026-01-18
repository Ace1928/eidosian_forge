from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_parse_no_bytes(self):
    deserializer = inventory_delta.InventoryDeltaDeserializer()
    err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, [])
    self.assertContainsRe(str(err), 'inventory delta is empty')