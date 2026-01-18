from breezy import osutils, tests
def test_view_on_commit_selected(self):
    wt = self.make_abc_tree_with_ab_view()
    self.run_bzr('add')
    out, err = self.run_bzr('commit -m "file in view" a')
    err_lines = err.splitlines()
    self.assertStartsWith(err_lines[0], 'Committing to:')
    self.assertEqual('added a', err_lines[1])
    self.assertEqual('Committed revision 1.', err_lines[2])
    self.assertEqual('', out)
    out, err = self.run_bzr('commit -m "file out of view" c', retcode=3)
    self.assertEqual('brz: ERROR: Specified file "c" is outside the current view: a, b\n', err)
    self.assertEqual('', out)