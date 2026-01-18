from breezy.tests import TestCaseWithTransport
from breezy.workingtree import WorkingTree
def test_switch_view(self):
    wt = self.make_branch_and_tree('.')
    self.run_bzr('view a b c')
    self.run_bzr('view e f --name foo')
    out, err = self.run_bzr('view --switch my')
    self.assertEqual(out, "Using 'my' view: a, b, c\n")
    out, err = self.run_bzr('view --switch off')
    self.assertEqual(out, "Disabled 'my' view.\n")
    out, err = self.run_bzr('view --switch off', retcode=3)
    self.assertContainsRe(err, 'No current view to disable')
    out, err = self.run_bzr('view --switch x --all', retcode=3)
    self.assertContainsRe(err, 'Both --switch and --all specified')