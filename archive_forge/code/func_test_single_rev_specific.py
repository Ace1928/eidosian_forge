import os
import stat
import time
from dulwich.objects import S_IFGITLINK, Blob, Tag, Tree
from dulwich.repo import Repo as GitRepo
from ... import osutils
from ...branch import Branch
from ...bzr import knit, versionedfile
from ...bzr.inventory import Inventory
from ...controldir import ControlDir
from ...repository import Repository
from ...tests import TestCaseWithTransport
from ..fetch import import_git_blob, import_git_submodule, import_git_tree
from ..mapping import DEFAULT_FILE_MODE, BzrGitMappingv1
from . import GitBranchBuilder
def test_single_rev_specific(self):
    path, gitsha = self.make_onerev_branch()
    oldrepo = self.open_git_repo(path)
    revid = oldrepo.get_mapping().revision_id_foreign_to_bzr(gitsha)
    newrepo = self.clone_git_repo(path, 'f', revision_id=revid)
    self.assertEqual([revid], newrepo.all_revision_ids())