from io import BytesIO
import breezy.bzr.xml5
from ... import errors, fifo_cache
from .. import inventory, serializer, xml6, xml7, xml8
from ..inventory import Inventory
from . import TestCase
def test_unpack_inventory_5a_cache_and_copy(self):
    entry_cache = fifo_cache.FIFOCache()
    inv = breezy.bzr.xml5.serializer_v5.read_inventory_from_lines(breezy.osutils.split_lines(_inventory_v5a), revision_id=b'test-rev-id', entry_cache=entry_cache, return_from_cache=False)
    for entry in inv.iter_just_entries():
        key = (entry.file_id, entry.revision)
        if entry.file_id is inv.root.file_id:
            self.assertFalse(key in entry_cache)
        else:
            self.assertIsNot(entry, entry_cache[key])