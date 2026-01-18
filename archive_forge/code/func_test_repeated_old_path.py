from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_repeated_old_path(self):
    inv = self.get_empty_inventory()
    file1 = inventory.InventoryFile(b'id1', 'path', inv.root.file_id)
    file1.revision = b'result'
    file1.text_size = 0
    file1.text_sha1 = b''
    file2 = inventory.InventoryFile(b'id2', 'path2', inv.root.file_id)
    file2.revision = b'result'
    file2.text_size = 0
    file2.text_sha1 = b''
    inv.add(file1)
    inv.add(file2)
    delta = [('path', None, b'id1', None), ('path', None, b'id2', None)]
    self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)