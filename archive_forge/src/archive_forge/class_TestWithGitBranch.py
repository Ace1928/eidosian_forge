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
class TestWithGitBranch(tests.TestCaseWithTransport):

    def setUp(self):
        tests.TestCaseWithTransport.setUp(self)
        r = dulwich.repo.Repo.create(self.test_dir)
        d = ControlDir.open(self.test_dir)
        self.git_branch = d.create_branch()

    def test_get_parent(self):
        self.assertIs(None, self.git_branch.get_parent())

    def test_get_stacked_on_url(self):
        self.assertRaises(UnstackableBranchFormat, self.git_branch.get_stacked_on_url)

    def test_get_physical_lock_status(self):
        self.assertFalse(self.git_branch.get_physical_lock_status())