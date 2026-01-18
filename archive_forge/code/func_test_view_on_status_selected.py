from breezy import osutils, tests
def test_view_on_status_selected(self):
    wt = self.make_abc_tree_with_ab_view()
    out, err = self.run_bzr('status a')
    self.assertEqual('', err)
    self.assertEqual('unknown:\n  a\n', out)
    out, err = self.run_bzr('status c', retcode=3)
    self.assertEqual('brz: ERROR: Specified file "c" is outside the current view: a, b\n', err)
    self.assertEqual('', out)