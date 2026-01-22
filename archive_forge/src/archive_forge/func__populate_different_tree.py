from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
def _populate_different_tree(tree, basis, delta):
    """Put all entries into tree, but at a unique location."""
    added_ids = set()
    added_paths = set()
    tree.add(['unique-dir'], ['directory'], [b'unique-dir-id'])
    for path, ie in basis.iter_entries_by_dir():
        if ie.file_id in added_ids:
            continue
        tree.add(['unique-dir/' + ie.file_id], [ie.kind], [ie.file_id])
        added_ids.add(ie.file_id)
    for old_path, new_path, file_id, ie in delta:
        if file_id in added_ids:
            continue
        tree.add(['unique-dir/' + file_id], [ie.kind], [file_id])