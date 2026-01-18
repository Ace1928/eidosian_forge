from breezy import osutils, tests
def test_view_on_merge(self):
    tree_1, tree_2 = self.make_abc_tree_and_clone_with_ab_view()
    out, err = self.run_bzr('merge -d tree_2 tree_1')
    self.assertEqualDiff("Operating on whole tree but only reporting on 'my' view.\n M  a\nAll changes applied successfully.\n", err)
    self.assertEqual('', out)