import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_set_root_id(self):

    def validate():
        with wt.lock_read():
            wt.current_dirstate()._validate()
    wt = self.make_workingtree('tree')
    wt.set_root_id(b'TREE-ROOTID')
    validate()
    wt.commit('somenthing')
    validate()
    wt.set_root_id(b'tree-rootid')
    validate()
    wt.commit('again')
    validate()