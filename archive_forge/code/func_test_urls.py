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
def test_urls(self):
    repo = self.make_repository('repo')
    self.assertIsInstance(repo.user_url, str)
    self.assertEqual(repo.user_url, repo.user_transport.base)
    self.assertEqual(repo.control_url.find(repo.user_url), 0)
    self.assertEqual(repo.control_url, repo.control_transport.base)