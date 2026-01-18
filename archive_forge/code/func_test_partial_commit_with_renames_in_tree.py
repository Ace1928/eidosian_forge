import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_partial_commit_with_renames_in_tree(self):
    t = self.make_branch_and_tree('.')
    self.build_tree(['dir/', 'dir/a', 'test'])
    t.add(['dir/', 'dir/a', 'test'])
    t.commit('initial commit')
    t.rename_one('dir/a', 'a')
    self.build_tree_contents([('test', b'changes in test')])
    out, err = self.run_bzr('commit test -m "partial commit"')
    self.assertEqual('', out)
    self.assertContainsRe(err, 'modified test\\nCommitted revision 2.')