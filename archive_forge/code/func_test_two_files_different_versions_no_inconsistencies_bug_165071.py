from breezy import errors
from breezy.bzr.tests.per_repository_vf import (
from breezy.bzr.tests.per_repository_vf.helpers import \
from breezy.tests import TestNotApplicable
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_two_files_different_versions_no_inconsistencies_bug_165071(self):
    """Two files, with different versions can be clean."""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['foo'])
    tree.smart_add(['.'])
    revid1 = tree.commit('1')
    self.build_tree(['bar'])
    tree.smart_add(['.'])
    revid2 = tree.commit('2')
    check_object = tree.branch.repository.check([revid1, revid2])
    check_object.report_results(verbose=True)
    self.assertContainsRe(self.get_log(), '0 unreferenced text versions')