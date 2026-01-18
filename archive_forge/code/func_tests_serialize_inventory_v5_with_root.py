from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def tests_serialize_inventory_v5_with_root(self):
    self.assertRoundTrips(_expected_inv_v5_root)