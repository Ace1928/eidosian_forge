from breezy import osutils, tests
def test_view_on_revert_selected(self):
    wt = self.make_abc_tree_with_ab_view()
    self.run_bzr('add')
    out, err = self.run_bzr('revert a')
    self.assertEqual('-   a\n', err)
    self.assertEqual('', out)
    out, err = self.run_bzr('revert c', retcode=3)
    self.assertEqual('brz: ERROR: Specified file "c" is outside the current view: a, b\n', err)
    self.assertEqual('', out)