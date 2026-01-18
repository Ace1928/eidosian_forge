from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_parse_unique_root_id_root_versioned_disabled(self):
    deserializer = inventory_delta.InventoryDeltaDeserializer()
    root_only_lines = b'format: bzr inventory delta v1 (bzr 1.14)\nparent: parent-id\nversion: a@e\xc3\xa5ample.com--2004\nversioned_root: false\ntree_references: true\nNone\x00/\x00an-id\x00\x00parent-id\x00dir\x00\x00\n'
    err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, osutils.split_lines(root_only_lines))
    self.assertContainsRe(str(err), 'Versioned root found')