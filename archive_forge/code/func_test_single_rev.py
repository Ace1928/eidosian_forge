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
def test_single_rev(self):
    path, gitsha = self.make_onerev_branch()
    oldrepo = Repository.open(path)
    revid = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha)
    self.assertEqual(gitsha, oldrepo._git.get_refs()[b'refs/heads/master'])
    newbranch = self.clone_git_branch(path, 'f')
    self.assertEqual([revid], newbranch.repository.all_revision_ids())