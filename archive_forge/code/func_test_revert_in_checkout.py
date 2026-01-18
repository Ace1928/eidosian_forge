import os
import breezy.osutils
from breezy.tests import TestCaseWithTransport
from breezy.trace import mutter
from breezy.workingtree import WorkingTree
def test_revert_in_checkout(self):
    os.mkdir('brach')
    os.chdir('brach')
    self._prepare_tree()
    self.run_bzr('checkout --lightweight . ../sprach')
    self.run_bzr('commit -m more')
    os.chdir('../sprach')
    self.assertEqual('', self.run_bzr('status')[0])
    self.run_bzr('revert')
    self.assertEqual('', self.run_bzr('status')[0])