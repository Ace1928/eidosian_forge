from .. import errors, tests, uncommit
def test_uncommit_remove_tags(self):
    tree, history = self.make_linear_tree()
    self.assertEqual(history[1], tree.last_revision())
    self.assertEqual((2, history[1]), tree.branch.last_revision_info())
    tree.branch.tags.set_tag('pointsatexisting', history[0])
    tree.branch.tags.set_tag('pointsatremoved', history[1])
    uncommit.uncommit(tree.branch, tree=tree)
    self.assertEqual(history[0], tree.last_revision())
    self.assertEqual((1, history[0]), tree.branch.last_revision_info())
    self.assertEqual({'pointsatexisting': history[0]}, tree.branch.tags.get_tag_dict())