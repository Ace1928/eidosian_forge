from breezy import osutils, tests
def test_view_on_diff_selected(self):
    wt = self.make_abc_tree_with_ab_view()
    self.run_bzr('add')
    out, err = self.run_bzr('diff a', retcode=1)
    self.assertEqual('', err)
    self.assertStartsWith(out, "=== added file 'a'\n")
    out, err = self.run_bzr('diff c', retcode=3)
    self.assertEqual('brz: ERROR: Specified file "c" is outside the current view: a, b\n', err)
    self.assertEqual('', out)