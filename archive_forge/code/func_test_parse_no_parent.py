from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_parse_no_parent(self):
    deserializer = inventory_delta.InventoryDeltaDeserializer()
    err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, [b'format: bzr inventory delta v1 (bzr 1.14)\n'])
    self.assertContainsRe(str(err), 'missing parent: marker')