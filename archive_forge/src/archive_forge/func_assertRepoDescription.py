import sys
from .. import branch as _mod_branch
from .. import controldir, errors, info
from .. import repository as _mod_repository
from .. import tests, workingtree
from ..bzr import branch as _mod_bzrbranch
def assertRepoDescription(self, format, expected=None):
    """Assert repository's format description matches expectations"""
    if expected is None:
        expected = format
    self.make_repository('%s_repo' % format, format=format)
    repo = _mod_repository.Repository.open('%s_repo' % format)
    self.assertEqual(expected, info.describe_format(repo.controldir, repo, None, None))