from breezy import errors, transport
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_unversion_renamed(self):
    tree = self.make_branch_and_tree('a')
    self.build_tree(['a/dir/', 'a/dir/f1', 'a/dir/f2', 'a/dir/f3', 'a/dir2/'])
    tree.add(['dir', 'dir/f1', 'dir/f2', 'dir/f3', 'dir2'])
    rev_id1 = tree.commit('init')
    revtree = tree.branch.repository.revision_tree(rev_id1)
    tree.rename_one('dir/f1', 'dir/a')
    tree.rename_one('dir/f2', 'dir/z')
    tree.move(['dir/f3'], 'dir2')
    self.assertThat(tree, HasPathRelations(revtree, [('', ''), ('dir/', 'dir/'), ('dir2/', 'dir2/'), ('dir/a', 'dir/f1'), ('dir/z', 'dir/f2'), ('dir2/f3', 'dir/f3')]))
    tree.unversion({'dir'})
    self.assertThat(tree, HasPathRelations(revtree, [('', ''), ('dir2/', 'dir2/'), ('dir2/f3', 'dir/f3')]))