import os
import dulwich
from dulwich.objects import Commit, Tag
from dulwich.repo import Repo as GitRepo
from ... import errors, revision, urlutils
from ...branch import Branch, InterBranch, UnstackableBranchFormat
from ...controldir import ControlDir
from ...repository import Repository
from .. import branch, tests
from ..dir import LocalGitControlDirFormat
from ..mapping import default_mapping
def test_open_by_ref(self):
    GitRepo.init('.')
    url = '{},ref={}'.format(urlutils.local_path_to_url(self.test_dir), urlutils.quote('refs/remotes/origin/unstable', safe=''))
    d = ControlDir.open(url)
    b = d.create_branch()
    self.assertEqual(b.ref, b'refs/remotes/origin/unstable')