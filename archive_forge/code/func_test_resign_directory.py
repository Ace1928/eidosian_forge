from breezy import gpg, tests
from breezy.bzr.testament import Testament
from breezy.controldir import ControlDir
def test_resign_directory(self):
    """Test --directory option"""
    wt = ControlDir.create_standalone_workingtree('a')
    a = wt.commit('base A', allow_pointless=True)
    b = wt.commit('base B', allow_pointless=True)
    c = wt.commit('base C', allow_pointless=True)
    repo = wt.branch.repository
    self.monkey_patch_gpg()
    self.run_bzr('re-sign --directory=a -r revid:' + a.decode('utf-8'))
    self.assertEqualSignature(repo, a)
    self.run_bzr('re-sign -d a %s' % b.decode('utf-8'))
    self.assertEqualSignature(repo, b)