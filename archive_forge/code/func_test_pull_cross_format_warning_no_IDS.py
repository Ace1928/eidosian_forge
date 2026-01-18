import os
import sys
from breezy import (branch, debug, osutils, tests, uncommit, urlutils,
from breezy.bzr import remote
from breezy.directory_service import directories
from breezy.tests import fixtures, script
def test_pull_cross_format_warning_no_IDS(self):
    """You get a warning for probably slow cross-format pulls.
        """
    debug.debug_flags.add('IDS_never')
    from_tree = self.make_branch_and_tree('from', format='2a')
    to_tree = self.make_branch_and_tree('to', format='1.14-rich-root')
    from_tree.commit(message='first commit')
    out, err = self.run_bzr(['pull', '-d', 'to', 'from'])
    self.assertContainsRe(err, '(?m)Doing on-the-fly conversion')