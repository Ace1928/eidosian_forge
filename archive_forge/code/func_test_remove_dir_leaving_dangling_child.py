from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_remove_dir_leaving_dangling_child(self):
    inv = self.get_empty_inventory()
    dir1 = inventory.InventoryDirectory(b'p-1', 'dir1', inv.root.file_id)
    dir1.revision = b'result'
    dir2 = inventory.InventoryDirectory(b'p-2', 'child1', b'p-1')
    dir2.revision = b'result'
    dir3 = inventory.InventoryDirectory(b'p-3', 'child2', b'p-1')
    dir3.revision = b'result'
    inv.add(dir1)
    inv.add(dir2)
    inv.add(dir3)
    delta = [('dir1', None, b'p-1', None), ('dir1/child2', None, b'p-3', None)]
    self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)