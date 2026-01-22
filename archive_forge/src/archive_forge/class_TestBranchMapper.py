from .... import tests
from .. import branch_mapper
from . import FastimportFeature
class TestBranchMapper(tests.TestCase):
    _test_needs_features = [FastimportFeature]

    def test_git_to_bzr(self):
        m = branch_mapper.BranchMapper()
        for git, bzr in {b'refs/heads/master': 'trunk', b'refs/heads/foo': 'foo', b'refs/tags/master': 'trunk.tag', b'refs/tags/foo': 'foo.tag', b'refs/remotes/origin/master': 'trunk.remote', b'refs/remotes/origin/foo': 'foo.remote'}.items():
            self.assertEqual(m.git_to_bzr(git), bzr)

    def test_git_to_bzr_with_slashes(self):
        m = branch_mapper.BranchMapper()
        for git, bzr in {b'refs/heads/master/slave': 'master/slave', b'refs/heads/foo/bar': 'foo/bar', b'refs/tags/master/slave': 'master/slave.tag', b'refs/tags/foo/bar': 'foo/bar.tag', b'refs/remotes/origin/master/slave': 'master/slave.remote', b'refs/remotes/origin/foo/bar': 'foo/bar.remote'}.items():
            self.assertEqual(m.git_to_bzr(git), bzr)

    def test_git_to_bzr_for_trunk(self):
        m = branch_mapper.BranchMapper()
        for git, bzr in {b'refs/heads/trunk': 'git-trunk', b'refs/tags/trunk': 'git-trunk.tag', b'refs/remotes/origin/trunk': 'git-trunk.remote', b'refs/heads/git-trunk': 'git-git-trunk', b'refs/tags/git-trunk': 'git-git-trunk.tag', b'refs/remotes/origin/git-trunk': 'git-git-trunk.remote'}.items():
            self.assertEqual(m.git_to_bzr(git), bzr)