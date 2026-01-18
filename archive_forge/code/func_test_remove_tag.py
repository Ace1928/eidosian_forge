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
def test_remove_tag(self):
    c1 = self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
    c2 = self.remote_real.do_commit(message=b'another commit', committer=b'committer <committer@example.com>', author=b'author <author@example.com>', ref=b'refs/tags/blah')
    remote = ControlDir.open(self.remote_url)
    remote_branch = remote.open_branch()
    remote_branch.tags.delete_tag('blah')
    self.assertRaises(NoSuchTag, remote_branch.tags.delete_tag, 'blah')
    self.assertEqual(self.remote_real.get_refs(), {b'refs/heads/master': self.remote_real.head(), b'HEAD': self.remote_real.head()})