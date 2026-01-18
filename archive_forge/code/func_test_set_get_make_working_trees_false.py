import re
from io import BytesIO
from ... import branch as _mod_branch
from ... import commit, controldir
from ... import delta as _mod_delta
from ... import errors, gpg, info, repository
from ... import revision as _mod_revision
from ... import tests, transport, upgrade, workingtree
from ...bzr import branch as _mod_bzrbranch
from ...bzr import inventory, knitpack_repo, remote
from ...bzr import repository as bzrrepository
from .. import per_repository, test_server
from ..matchers import *
def test_set_get_make_working_trees_false(self):
    repo = self.make_repository('repo')
    try:
        repo.set_make_working_trees(False)
    except (errors.RepositoryUpgradeRequired, errors.UnsupportedOperation) as e:
        raise tests.TestNotApplicable('Format does not support this flag.')
    self.assertFalse(repo.make_working_trees())