import doctest
import os
import re
import sys
from testtools.matchers import DocTestMatches
from ... import config, ignores, msgeditor, osutils
from ...controldir import ControlDir
from .. import TestCaseWithTransport, features, test_foreign
from ..test_bedding import override_whoami
def test_strict_commit_no_changes(self):
    """commit --strict gives "no changes" if there is nothing to commit"""
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/a'])
    tree.add('a')
    tree.commit('adding a')
    self.run_bzr_error(['No changes to commit'], 'commit --strict -m no-changes', working_dir='tree')
    self.run_bzr('commit --strict --unchanged -m no-changes', working_dir='tree')