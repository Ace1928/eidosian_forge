from breezy import osutils, tests
def test_view_on_commit(self):
    wt = self.make_abc_tree_with_ab_view()
    self.run_bzr('add')
    out, err = self.run_bzr('commit -m "testing commit"')
    err_lines = err.splitlines()
    self.assertEqual('Ignoring files outside view. View is a, b', err_lines[0])
    self.assertStartsWith(err_lines[1], 'Committing to:')
    self.assertIn('added a', [err_lines[2], err_lines[3]])
    self.assertIn('added b', [err_lines[2], err_lines[3]])
    self.assertEqual('Committed revision 1.', err_lines[4])
    self.assertEqual('', out)