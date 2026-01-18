from breezy import osutils, tests
def test_view_on_status(self):
    wt = self.make_abc_tree_with_ab_view()
    out, err = self.run_bzr('status')
    self.assertEqual('Ignoring files outside view. View is a, b\n', err)
    self.assertEqual('unknown:\n  a\n  b\n', out)