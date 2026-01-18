from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
def test_rename(self):
    inv = self.make_init_inventory()
    inv = inv.create_by_apply_delta([(None, 'a', b'a-id', self.make_file(b'a-id', 'a', b'tree-root'))], b'new-rev-1')
    self.assertEqual('a', inv.id2path(b'a-id'))
    a_ie = inv.get_entry(b'a-id')
    b_ie = self.make_file(a_ie.file_id, 'b', a_ie.parent_id)
    inv = inv.create_by_apply_delta([('a', 'b', b'a-id', b_ie)], b'new-rev-2')
    self.assertEqual('b', inv.id2path(b'a-id'))