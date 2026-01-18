from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_reconcile_behaviour(self):
    """Populate a repository and reconcile it, verifying the state before
        and after.
        """
    repo, scenario = self.prepare_test_repository()
    with repo.lock_read():
        self.assertParentsMatch(scenario.populated_parents(), repo, b'before')
        vf_shas = self.shas_for_versions_of_file(repo, scenario.all_versions_after_reconcile())
    result = repo.reconcile(thorough=True)
    with repo.lock_read():
        self.assertParentsMatch(scenario.corrected_parents(), repo, b'after')
        self.assertEqual(vf_shas, self.shas_for_versions_of_file(repo, scenario.all_versions_after_reconcile()))
        for file_version in scenario.corrected_fulltexts():
            key = (b'a-file-id', file_version)
            self.assertEqual({key: ()}, repo.texts.get_parent_map([key]))
            self.assertIsInstance(next(repo.texts.get_record_stream([key], 'unordered', True)).get_bytes_as('fulltext'), bytes)