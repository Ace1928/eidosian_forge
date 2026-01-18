from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_parse_unversioned_root_versioning_enabled(self):
    deserializer = inventory_delta.InventoryDeltaDeserializer()
    parse_result = deserializer.parse_text_bytes(osutils.split_lines(root_only_unversioned))
    expected_entry = inventory.make_entry('directory', '', None, b'TREE_ROOT')
    expected_entry.revision = b'entry-version'
    self.assertEqual((b'null:', b'entry-version', False, False, [(None, '', b'TREE_ROOT', expected_entry)]), parse_result)