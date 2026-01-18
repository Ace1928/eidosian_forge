from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def test_repeated_new_path(self):
    inv = self.get_empty_inventory()
    file1 = inventory.InventoryFile(b'id1', 'path', inv.root.file_id)
    file1.revision = b'result'
    file1.text_size = 0
    file1.text_sha1 = b''
    file2 = file1.copy()
    file2.file_id = b'id2'
    delta = [(None, 'path', b'id1', file1), (None, 'path', b'id2', file2)]
    self.assertRaises(errors.InconsistentDelta, self.apply_delta, self, inv, delta)