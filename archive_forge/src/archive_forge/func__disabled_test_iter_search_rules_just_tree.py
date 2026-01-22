from breezy import rules
from breezy.tests.per_tree import TestCaseWithTree
def _disabled_test_iter_search_rules_just_tree(self):
    per_user = self.make_per_user_searcher('')
    tree = self.make_tree_with_rules('[name ./a.txt]\nfoo=baz\n[name *.txt]\nfoo=bar\na=True\n')
    result = list(tree.iter_search_rules(['a.txt', 'dir/a.txt'], _default_searcher=per_user))
    self.assertEqual((('foo', 'baz'),), result[0])
    self.assertEqual((('foo', 'bar'), ('a', 'True')), result[1])