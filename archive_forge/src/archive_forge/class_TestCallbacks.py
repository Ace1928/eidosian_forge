from breezy import errors
from breezy.bzr.tests.per_repository_vf import (
from breezy.bzr.tests.per_repository_vf.helpers import \
from breezy.tests import TestNotApplicable
from breezy.tests.scenarios import load_tests_apply_scenarios
class TestCallbacks(TestCaseWithRepository):
    scenarios = all_repository_vf_format_scenarios()

    def test_callback_tree_and_branch(self):
        tree = self.make_branch_and_tree('foo')
        revid = tree.commit('foo')
        tree.lock_read()
        self.addCleanup(tree.unlock)
        needed_refs = {}
        for ref in tree._get_check_refs():
            needed_refs.setdefault(ref, []).append(tree)
        for ref in tree.branch._get_check_refs():
            needed_refs.setdefault(ref, []).append(tree.branch)
        self.tree_check = tree._check
        self.branch_check = tree.branch.check
        self.overrideAttr(tree, '_check', self.tree_callback)
        self.overrideAttr(tree.branch, 'check', self.branch_callback)
        self.callbacks = []
        tree.branch.repository.check([revid], callback_refs=needed_refs)
        self.assertNotEqual([], self.callbacks)

    def tree_callback(self, refs):
        self.callbacks.append(('tree', refs))
        return self.tree_check(refs)

    def branch_callback(self, refs):
        self.callbacks.append(('branch', refs))
        return self.branch_check(refs)