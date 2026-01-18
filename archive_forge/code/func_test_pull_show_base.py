import os
import sys
from breezy import (branch, debug, osutils, tests, uncommit, urlutils,
from breezy.bzr import remote
from breezy.directory_service import directories
from breezy.tests import fixtures, script
def test_pull_show_base(self):
    """brz pull supports --show-base

        see https://bugs.launchpad.net/bzr/+bug/202374"""
    a_tree = self.example_branch('a')
    b_tree = a_tree.controldir.sprout('b').open_workingtree()
    with open(osutils.pathjoin('a', 'hello'), 'w') as f:
        f.write('fee')
    a_tree.commit('fee')
    with open(osutils.pathjoin('b', 'hello'), 'w') as f:
        f.write('fie')
    out, err = self.run_bzr(['pull', '-d', 'b', 'a', '--show-base'])
    self.assertEqual(err, ' M  hello\nText conflict in hello\n1 conflicts encountered.\n')
    with open(osutils.pathjoin('b', 'hello')) as f:
        self.assertEqualDiff('<<<<<<< TREE\nfie||||||| BASE-REVISION\nfoo=======\nfee>>>>>>> MERGE-SOURCE\n', f.read())