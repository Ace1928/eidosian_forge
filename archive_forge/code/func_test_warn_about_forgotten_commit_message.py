import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_warn_about_forgotten_commit_message(self):
    """Test that the lack of -m parameter is caught"""
    wt = self.make_branch_and_tree('.')
    self.build_tree(['one', 'two'])
    wt.add(['two'])
    out, err = self.run_bzr('commit -m one two')
    self.assertContainsRe(err, 'The commit message is a file name')