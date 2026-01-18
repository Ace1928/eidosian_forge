from breezy import gpg, tests
from breezy.bzr.testament import Testament
from breezy.controldir import ControlDir
def test_resign_multiple(self):
    wt, rs = self.setup_tree()
    repo = wt.branch.repository
    self.monkey_patch_gpg()
    self.run_bzr('re-sign ' + ' '.join((r.decode('utf-8') for r in rs)))
    for r in rs:
        self.assertEqualSignature(repo, r)