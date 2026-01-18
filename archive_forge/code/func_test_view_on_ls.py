from breezy import osutils, tests
def test_view_on_ls(self):
    wt = self.make_abc_tree_with_ab_view()
    self.run_bzr('add')
    out, err = self.run_bzr('ls')
    out_lines = out.splitlines()
    self.assertEqual('Ignoring files outside view. View is a, b\n', err)
    self.assertEqual('a', out_lines[0])
    self.assertEqual('b', out_lines[1])