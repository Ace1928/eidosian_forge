from ...tests import TestCase
from ..urls import git_url_to_bzr_url
def test_with_ref(self):
    self.assertEqual(git_url_to_bzr_url('foo:bar/path', ref=b'HEAD'), 'git+ssh://foo/bar/path')
    self.assertEqual(git_url_to_bzr_url('foo:bar/path', ref=b'refs/heads/blah'), 'git+ssh://foo/bar/path,branch=blah')
    self.assertEqual(git_url_to_bzr_url('foo:bar/path', ref=b'refs/tags/blah'), 'git+ssh://foo/bar/path,ref=refs%2Ftags%2Fblah')