import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_set_commit_message(self):
    msgeditor.hooks.install_named_hook('set_commit_message', lambda commit_obj, msg: 'save me some typing\n', None)
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/hello.txt'])
    tree.add('hello.txt')
    out, err = self.run_bzr('commit tree/hello.txt')
    last_rev = tree.branch.repository.get_revision(tree.last_revision())
    self.assertEqual('save me some typing\n', last_rev.message)