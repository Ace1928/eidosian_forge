import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_parent_child_swap(self):
    state = self.assertUpdate(active=[('A/', b'A-id'), ('A/B/', b'B-id'), ('A/B/C', b'C-id')], basis=[('A/', b'A-id'), ('A/B/', b'B-id'), ('A/B/C', b'C-id')], target=[('A/', b'B-id'), ('A/B/', b'A-id'), ('A/B/C', b'C-id')])
    state = self.assertUpdate(active=[('A/', b'B-id'), ('A/B/', b'A-id'), ('A/B/C', b'C-id')], basis=[('A/', b'A-id'), ('A/B/', b'B-id'), ('A/B/C', b'C-id')], target=[('A/', b'B-id'), ('A/B/', b'A-id'), ('A/B/C', b'C-id')])
    state = self.assertUpdate(active=[], basis=[('A/', b'A-id'), ('A/B/', b'B-id'), ('A/B/C', b'C-id')], target=[('A/', b'B-id'), ('A/B/', b'A-id'), ('A/B/C', b'C-id')])
    state = self.assertUpdate(active=[('D/', b'A-id'), ('D/E/', b'B-id'), ('F', b'C-id')], basis=[('A/', b'A-id'), ('A/B/', b'B-id'), ('A/B/C', b'C-id')], target=[('A/', b'B-id'), ('A/B/', b'A-id'), ('A/B/C', b'C-id')])