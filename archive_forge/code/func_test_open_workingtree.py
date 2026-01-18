import os
from dulwich.repo import Repo as GitRepo
from ... import controldir, errors, urlutils
from ...tests import TestSkipped
from ...transport import get_transport
from .. import dir, tests, workingtree
def test_open_workingtree(self):
    r = GitRepo.init('.')
    r.do_commit(message=b'message')
    gd = controldir.ControlDir.open('.')
    wt = gd.open_workingtree()
    self.assertIsInstance(wt, workingtree.GitWorkingTree)