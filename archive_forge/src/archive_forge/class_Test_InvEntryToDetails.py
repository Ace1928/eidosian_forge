import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
class Test_InvEntryToDetails(tests.TestCase):

    def assertDetails(self, expected, inv_entry):
        details = dirstate.DirState._inv_entry_to_details(inv_entry)
        self.assertEqual(expected, details)
        minikind, fingerprint, size, executable, tree_data = details
        self.assertIsInstance(minikind, bytes)
        self.assertIsInstance(fingerprint, bytes)
        self.assertIsInstance(tree_data, bytes)

    def test_unicode_symlink(self):
        inv_entry = inventory.InventoryLink(b'link-file-id', 'nam€e', b'link-parent-id')
        inv_entry.revision = b'link-revision-id'
        target = 'link-targ€t'
        inv_entry.symlink_target = target
        self.assertDetails((b'l', target.encode('UTF-8'), 0, False, b'link-revision-id'), inv_entry)