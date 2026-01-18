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
def test_push_diverged(self):
    c1 = self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>', ref=b'refs/heads/newbranch')
    remote = ControlDir.open(self.remote_url)
    wt = self.make_branch_and_tree('local', format=self._from_format)
    self.build_tree(['local/blah'])
    wt.add(['blah'])
    revid = wt.commit('blah')
    newbranch = remote.open_branch('newbranch')
    if self._from_format == 'git':
        self.assertRaises(DivergedBranches, wt.branch.push, newbranch)
    else:
        self.assertRaises(DivergedBranches, wt.branch.push, newbranch, lossy=True)
    self.assertEqual({b'refs/heads/newbranch': c1}, self.remote_real.get_refs())
    if self._from_format == 'git':
        wt.branch.push(newbranch, overwrite=True)
    else:
        wt.branch.push(newbranch, lossy=True, overwrite=True)
    self.assertNotEqual(c1, self.remote_real.refs[b'refs/heads/newbranch'])