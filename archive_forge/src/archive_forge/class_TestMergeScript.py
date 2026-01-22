import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
class TestMergeScript(script.TestCaseWithTransportAndScript):

    def test_merge_empty_branch(self):
        source = self.make_branch_and_tree('source')
        self.build_tree(['source/a'])
        source.add('a')
        source.commit('Added a', rev_id=b'rev1')
        target = self.make_branch_and_tree('target')
        self.run_script('$ brz merge -d target source\n2>brz: ERROR: Merging into empty branches not currently supported, https://bugs.launchpad.net/bzr/+bug/308562\n')