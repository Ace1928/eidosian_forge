from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_unversioned_root(self):
    old_inv = Inventory(None)
    new_inv = Inventory(None)
    root = new_inv.make_entry('directory', '', None, b'TREE_ROOT')
    root.revision = b'entry-version'
    new_inv.add(root)
    delta = new_inv._make_delta(old_inv)
    serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=False, tree_references=False)
    serialized_lines = serializer.delta_to_lines(NULL_REVISION, b'entry-version', delta)
    self.assertEqual(BytesIO(root_only_unversioned).readlines(), serialized_lines)
    deserializer = inventory_delta.InventoryDeltaDeserializer()
    self.assertEqual((NULL_REVISION, b'entry-version', False, False, delta), deserializer.parse_text_bytes(serialized_lines))