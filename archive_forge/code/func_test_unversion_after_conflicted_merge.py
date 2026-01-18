from breezy import errors, transport
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_unversion_after_conflicted_merge(self):
    tree_a = self.make_branch_and_tree('A')
    self.build_tree(['A/a/', 'A/a/m', 'A/a/n'])
    tree_a.add(['a', 'a/m', 'a/n'])
    tree_a.commit('init')
    tree_b = tree_a.controldir.sprout('B').open_workingtree()
    self.build_tree(['B/xyz/'])
    tree_b.add(['xyz'])
    tree_b.rename_one('a/m', 'xyz/m')
    tree_b.unversion(['a'])
    tree_b.commit('delete in B')
    self.assertThat(tree_b, HasPathRelations(tree_a, [('', ''), ('xyz/', None), ('xyz/m', 'a/m')]))
    self.build_tree_contents([('A/a/n', b'new contents for n\n')])
    tree_a.commit('change n in A')
    conflicts = tree_b.merge_from_branch(tree_a.branch)
    if tree_b.has_versioned_directories():
        self.assertEqual(4, len(conflicts))
    else:
        self.assertEqual(1, len(conflicts))
    self.assertThat(tree_b, HasPathRelations(tree_a, [('', ''), ('a/', 'a/'), ('xyz/', None), ('a/n.OTHER', 'a/n'), ('xyz/m', 'a/m')]))
    tree_b.unversion(['a'])
    self.assertThat(tree_b, HasPathRelations(tree_a, [('', ''), ('xyz/', None), ('xyz/m', 'a/m')]))