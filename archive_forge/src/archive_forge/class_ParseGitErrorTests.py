import gzip
import os
import time
from io import BytesIO
from dulwich import porcelain
from dulwich.errors import HangupException
from dulwich.repo import Repo as GitRepo
from ...branch import Branch
from ...controldir import BranchReferenceLoop, ControlDir
from ...errors import (ConnectionReset, DivergedBranches, NoSuchTag,
from ...tests import TestCase, TestCaseWithTransport
from ...tests.features import ExecutableFeature
from ...urlutils import join as urljoin
from ..mapping import default_mapping
from ..remote import (GitRemoteRevisionTree, GitSmartRemoteNotSupported,
from ..tree import MissingNestedTree
class ParseGitErrorTests(TestCase):

    def test_unknown(self):
        e = parse_git_error('url', 'foo')
        self.assertIsInstance(e, RemoteGitError)

    def test_connection_closed(self):
        e = parse_git_error('url', 'The remote server unexpectedly closed the connection.')
        self.assertIsInstance(e, TransportError)

    def test_notbrancherror(self):
        e = parse_git_error('url', '\n Could not find Repository foo/bar')
        self.assertIsInstance(e, NotBranchError)

    def test_notbrancherror_launchpad(self):
        e = parse_git_error('url', "Repository 'foo/bar' not found.")
        self.assertIsInstance(e, NotBranchError)

    def test_notbrancherror_github(self):
        e = parse_git_error('url', 'Repository not found.\n')
        self.assertIsInstance(e, NotBranchError)

    def test_notbrancherror_normal(self):
        e = parse_git_error('url', "fatal: '/srv/git/lintian-brush' does not appear to be a git repository")
        self.assertIsInstance(e, NotBranchError)

    def test_head_update(self):
        e = parse_git_error('url', 'HEAD failed to update\n')
        self.assertIsInstance(e, HeadUpdateFailed)

    def test_permission_dnied(self):
        e = parse_git_error('url', 'access denied or repository not exported: /debian/altermime.git')
        self.assertIsInstance(e, PermissionDenied)

    def test_permission_denied_gitlab(self):
        e = parse_git_error('url', 'GitLab: You are not allowed to push code to this project.\n')
        self.assertIsInstance(e, PermissionDenied)

    def test_permission_denied_github(self):
        e = parse_git_error('url', 'Permission to porridge/gaduhistory.git denied to jelmer.')
        self.assertIsInstance(e, PermissionDenied)
        self.assertEqual(e.path, 'porridge/gaduhistory.git')
        self.assertEqual(e.extra, ': denied to jelmer')

    def test_pre_receive_hook_declined(self):
        e = parse_git_error('url', 'pre-receive hook declined')
        self.assertIsInstance(e, PermissionDenied)
        self.assertEqual(e.path, 'url')
        self.assertEqual(e.extra, ': pre-receive hook declined')

    def test_invalid_repo_name(self):
        e = parse_git_error('url', 'Gregwar/fatcat/tree/debian is not a valid repository name\nEmail support@github.com for help\n')
        self.assertIsInstance(e, NotBranchError)

    def test_invalid_git_error(self):
        self.assertEqual(PermissionDenied('url', 'GitLab: You are not allowed to push code to protected branches on this project.'), parse_git_error('url', RemoteGitError('GitLab: You are not allowed to push code to protected branches on this project.')))

    def test_protected_branch(self):
        self.assertEqual(ProtectedBranchHookDeclined(msg='protected branch hook declined'), parse_git_error('url', RemoteGitError('protected branch hook declined')))

    def test_host_key_verification(self):
        self.assertEqual(TransportError('Host key verification failed'), parse_git_error('url', RemoteGitError('Host key verification failed.')))

    def test_connection_reset_by_peer(self):
        self.assertEqual(ConnectionReset('[Errno 104] Connection reset by peer'), parse_git_error('url', RemoteGitError('[Errno 104] Connection reset by peer')))

    def test_http_unexpected(self):
        self.assertEqual(UnexpectedHttpStatus('https://example.com/bigint.git/git-upload-pack', 403, extra='unexpected http resp 403 for https://example.com/bigint.git/git-upload-pack'), parse_git_error('url', RemoteGitError('unexpected http resp 403 for https://example.com/bigint.git/git-upload-pack')))