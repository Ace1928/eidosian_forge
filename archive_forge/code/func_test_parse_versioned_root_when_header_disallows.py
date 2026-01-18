from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
def test_parse_versioned_root_when_header_disallows(self):
    deserializer = inventory_delta.InventoryDeltaDeserializer()
    lines = b'format: bzr inventory delta v1 (bzr 1.14)\nparent: null:\nversion: entry-version\nversioned_root: false\ntree_references: false\nNone\x00/\x00TREE_ROOT\x00\x00a@e\xc3\xa5ample.com--2004\x00dir\n'
    err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, osutils.split_lines(lines))
    self.assertContainsRe(str(err), 'Versioned root found')