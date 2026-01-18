import os
from breezy import config, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests
from breezy.bzr import inventorytree
from breezy.bzr.inventorytree import InventoryTreeChange
from breezy.tests import features, per_repository
from ..test_bedding import override_whoami
def test_stacked_repositories_reject_commit_builder(self):
    repo_basis = self.make_repository('basis')
    branch = self.make_branch('local')
    repo_local = branch.repository
    try:
        repo_local.add_fallback_repository(repo_basis)
    except errors.UnstackableRepositoryFormat:
        raise tests.TestNotApplicable('not a stackable format.')
    self.addCleanup(repo_local.lock_write().unlock)
    if not repo_local._format.supports_chks:
        self.assertRaises(errors.BzrError, repo_local.get_commit_builder, branch, [], branch.get_config_stack())
    else:
        builder = repo_local.get_commit_builder(branch, [], branch.get_config_stack())
        builder.abort()