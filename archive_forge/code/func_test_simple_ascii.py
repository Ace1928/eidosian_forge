from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_simple_ascii(self):
    val = breezy.bzr.xml_serializer.encode_and_escape('foo bar')
    self.assertEqual(b'foo bar', val)
    val2 = breezy.bzr.xml_serializer.encode_and_escape('foo bar')
    self.assertIs(val2, val)