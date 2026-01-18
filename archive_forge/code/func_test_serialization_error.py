from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_serialization_error(self):
    s_v5 = breezy.bzr.xml5.serializer_v5
    e = self.assertRaises(serializer.UnexpectedInventoryFormat, s_v5.read_inventory_from_lines, [b'<Notquitexml'])
    self.assertEqual(str(e), 'unclosed token: line 1, column 0')