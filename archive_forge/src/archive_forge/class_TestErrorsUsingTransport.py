import inspect
import re
import socket
import sys
from .. import controldir, errors, osutils, tests, urlutils
class TestErrorsUsingTransport(tests.TestCaseWithMemoryTransport):
    """Tests for errors that need to use a branch or repo."""

    def test_no_public_branch(self):
        b = self.make_branch('.')
        error = errors.NoPublicBranch(b)
        url = urlutils.unescape_for_display(b.base, 'ascii')
        self.assertEqualDiff('There is no public branch set for "%s".' % url, str(error))

    def test_no_repo(self):
        dir = controldir.ControlDir.create(self.get_url())
        error = errors.NoRepositoryPresent(dir)
        self.assertNotEqual(-1, str(error).find(dir.transport.clone('..').base))
        self.assertEqual(-1, str(error).find(dir.transport.base))

    def test_corrupt_repository(self):
        repo = self.make_repository('.')
        error = errors.CorruptRepository(repo)
        self.assertEqualDiff('An error has been detected in the repository %s.\nPlease run brz reconcile on this repository.' % repo.controldir.root_transport.base, str(error))

    def test_not_branch_bzrdir_with_repo(self):
        controldir = self.make_repository('repo').controldir
        err = errors.NotBranchError('path', controldir=controldir)
        self.assertEqual('Not a branch: "path": location is a repository.', str(err))

    def test_not_branch_bzrdir_without_repo(self):
        controldir = self.make_controldir('bzrdir')
        err = errors.NotBranchError('path', controldir=controldir)
        self.assertEqual('Not a branch: "path".', str(err))

    def test_not_branch_laziness(self):
        real_bzrdir = self.make_controldir('path')

        class FakeBzrDir:

            def __init__(self):
                self.calls = []

            def open_repository(self):
                self.calls.append('open_repository')
                raise errors.NoRepositoryPresent(real_bzrdir)
        fake_bzrdir = FakeBzrDir()
        err = errors.NotBranchError('path', controldir=fake_bzrdir)
        self.assertEqual([], fake_bzrdir.calls)
        str(err)
        self.assertEqual(['open_repository'], fake_bzrdir.calls)
        str(err)
        self.assertEqual(['open_repository'], fake_bzrdir.calls)