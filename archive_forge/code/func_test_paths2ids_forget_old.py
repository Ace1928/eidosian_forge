from breezy import errors, tests
from breezy.tests.per_tree import TestCaseWithTree
def test_paths2ids_forget_old(self):
    work_tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/file'])
    work_tree.add('file')
    work_tree.commit('commit old state')
    work_tree.remove('file')
    if not work_tree.supports_setting_file_ids():
        raise tests.TestNotApplicable('test not applicable on non-inventory tests')
    tree = self._convert_tree(work_tree)
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertEqual(set(), tree.paths2ids(['file'], require_versioned=False))