from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_parse_tree_when_disabled(self):
    deserializer = inventory_delta.InventoryDeltaDeserializer(allow_tree_references=False)
    err = self.assertRaises(inventory_delta.IncompatibleInventoryDelta, deserializer.parse_text_bytes, osutils.split_lines(reference_lines))
    self.assertEqual('Tree reference not allowed', str(err))