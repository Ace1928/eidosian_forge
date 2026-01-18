from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_parse_special_revid_not_valid_last_mod(self):
    deserializer = inventory_delta.InventoryDeltaDeserializer()
    root_only_lines = b'format: bzr inventory delta v1 (bzr 1.14)\nparent: null:\nversion: null:\nversioned_root: false\ntree_references: true\nNone\x00/\x00TREE_ROOT\x00\x00null:\x00dir\x00\x00\n'
    err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, osutils.split_lines(root_only_lines))
    self.assertContainsRe(str(err), 'special revisionid found')