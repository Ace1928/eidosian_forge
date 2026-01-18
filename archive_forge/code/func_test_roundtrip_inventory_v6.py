from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_roundtrip_inventory_v6(self):
    inv = self.get_sample_inventory()
    lines = xml6.serializer_v6.write_inventory_to_lines(inv)
    self.assertEqualDiff(_expected_inv_v6, b''.join(lines))
    inv2 = xml6.serializer_v6.read_inventory_from_lines(lines)
    self.assertEqual(4, len(inv2))
    for path, ie in inv.iter_entries():
        self.assertEqual(ie, inv2.get_entry(ie.file_id))