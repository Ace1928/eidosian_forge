from breezy import branch, controldir, errors, revision
from breezy.tests import TestNotApplicable, fixtures, per_branch
def test_pull_overwrite_set_tags(self):
    tree_a = self.make_branch_and_tree('tree_a')
    if not tree_a.branch.supports_tags():
        raise TestNotApplicable('branch does not support tags')
    rev1 = tree_a.commit('message 1')
    tree_a.branch.tags.set_tag('tag1', rev1)
    tree_b = tree_a.controldir.sprout('tree_b').open_workingtree()
    rev2b = tree_b.commit('message 2b')
    tree_b.branch.tags.set_tag('tag1', rev2b)
    rev1b = tree_a.commit('message 1b')
    tree_a.branch.get_config_stack().set('branch.fetch_tags', True)
    self.assertRaises(errors.DivergedBranches, tree_a.pull, tree_b.branch)
    self.assertRaises(errors.DivergedBranches, tree_a.branch.pull, tree_b.branch, overwrite=set(), stop_revision=rev2b)
    self.assertEqual(rev1b, tree_a.branch.last_revision())
    self.assertEqual(tree_a.branch.tags.get_tag_dict(), {'tag1': rev1})
    if tree_a.branch.repository._format.supports_unreferenced_revisions:
        self.assertTrue(tree_a.branch.repository.has_revision(rev2b))
    tree_a.branch.pull(tree_b.branch, overwrite={'history'}, stop_revision=rev2b)
    self.assertEqual(rev2b, tree_a.branch.last_revision())
    self.assertEqual(tree_b.branch.last_revision(), tree_a.branch.last_revision())
    self.assertEqual(rev1, tree_a.branch.tags.lookup_tag('tag1'))
    tree_a.branch.pull(tree_b.branch, overwrite={'history', 'tags'}, stop_revision=rev2b)
    self.assertEqual(rev2b, tree_a.branch.tags.lookup_tag('tag1'))