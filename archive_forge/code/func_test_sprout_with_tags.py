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
def test_sprout_with_tags(self):
    c1 = self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
    c2 = self.remote_real.do_commit(message=b'another commit', committer=b'committer <committer@example.com>', author=b'author <author@example.com>', ref=b'refs/tags/another')
    self.remote_real.refs[b'refs/tags/blah'] = self.remote_real.head()
    remote = ControlDir.open(self.remote_url)
    self.make_controldir('local', format=self._to_format)
    local = remote.sprout('local')
    local_branch = local.open_branch()
    self.assertEqual(default_mapping.revision_id_foreign_to_bzr(c1), local_branch.last_revision())
    self.assertEqual({'blah': local_branch.last_revision(), 'another': default_mapping.revision_id_foreign_to_bzr(c2)}, local_branch.tags.get_tag_dict())