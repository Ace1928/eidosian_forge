import os
from dulwich.repo import Repo as GitRepo
from ... import controldir, errors, urlutils
from ...tests import TestSkipped
from ...transport import get_transport
from .. import dir, tests, workingtree
def test_open_workingtree_bare(self):
    GitRepo.init_bare('.')
    gd = controldir.ControlDir.open('.')
    self.assertRaises(errors.NoWorkingTree, gd.open_workingtree)