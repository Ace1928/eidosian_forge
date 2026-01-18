from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_parse_bad_format(self):
    deserializer = inventory_delta.InventoryDeltaDeserializer()
    err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, [b'format: foo\n'])
    self.assertContainsRe(str(err), 'unknown format')