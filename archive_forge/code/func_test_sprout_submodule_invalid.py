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
def test_sprout_submodule_invalid(self):
    self.sub_real = GitRepo.init('sub', mkdir=True)
    self.sub_real.do_commit(message=b'message in sub', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
    self.sub_real.clone('remote/nested')
    self.remote_real.stage('nested')
    self.permit_url(urljoin(self.remote_url, '../sub'))
    self.assertIn(b'nested', self.remote_real.open_index())
    self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
    remote = ControlDir.open(self.remote_url)
    self.make_controldir('local', format=self._to_format)
    local = remote.sprout('local')
    self.assertEqual(default_mapping.revision_id_foreign_to_bzr(self.remote_real.head()), local.open_branch().last_revision())
    self.assertRaises(MissingNestedTree, local.open_workingtree().get_nested_tree, 'nested')