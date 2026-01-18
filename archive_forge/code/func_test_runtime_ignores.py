from breezy import bedding, ignores, tests
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_runtime_ignores(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('.bzrignore', b'')])
    ignores._set_user_ignores([])
    orig_runtime = ignores._runtime_ignores
    try:
        ignores._runtime_ignores = set()
        self.assertEqual(None, tree.is_ignored('foobar.py'))
        tree._flush_ignore_list_cache()
        ignores.add_runtime_ignores(['./foobar.py'])
        self.assertEqual({'./foobar.py'}, ignores.get_runtime_ignores())
        self.assertEqual('./foobar.py', tree.is_ignored('foobar.py'))
    finally:
        ignores._runtime_ignores = orig_runtime