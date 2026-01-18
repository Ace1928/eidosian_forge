import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_verbose_commit_includes_master_location(self):
    """Location of master is displayed when committing to bound branch"""
    a_tree = self.make_branch_and_tree('a')
    self.build_tree(['a/b'])
    a_tree.add('b')
    a_tree.commit(message='Initial message')
    b_tree = a_tree.branch.create_checkout('b')
    expected = '{}/'.format(osutils.abspath('a'))
    out, err = self.run_bzr('commit -m blah --unchanged', working_dir='b')
    self.assertEqual(err, 'Committing to: %s\nCommitted revision 2.\n' % expected)