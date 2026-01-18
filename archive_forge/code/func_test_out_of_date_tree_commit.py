import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_out_of_date_tree_commit(self):
    tree = self.make_branch_and_tree('branch')
    checkout = tree.branch.create_checkout('checkout', lightweight=True)
    tree.commit('message branch', allow_pointless=True)
    output = self.run_bzr('commit --unchanged -m checkout_message checkout', retcode=3)
    self.assertEqual(output, ('', "brz: ERROR: Working tree is out of date, please run 'brz update'.\n"))