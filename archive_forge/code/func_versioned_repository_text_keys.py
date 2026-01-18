from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
def versioned_repository_text_keys(self):
    return {(b'TREE_ROOT', b'broken-revision-1-2'): [(b'TREE_ROOT', b'parent-1'), (b'TREE_ROOT', b'parent-2')], (b'TREE_ROOT', b'broken-revision-2-1'): [(b'TREE_ROOT', b'parent-2'), (b'TREE_ROOT', b'parent-1')], (b'TREE_ROOT', b'parent-1'): [NULL_REVISION], (b'TREE_ROOT', b'parent-2'): [NULL_REVISION]}