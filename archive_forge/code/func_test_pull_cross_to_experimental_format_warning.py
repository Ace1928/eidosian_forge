import os
import sys
from breezy import (branch, debug, osutils, tests, uncommit, urlutils,
from breezy.bzr import remote
from breezy.directory_service import directories
from breezy.tests import fixtures, script
def test_pull_cross_to_experimental_format_warning(self):
    """You get a warning for pulling into experimental formats.
        """
    from_tree = self.make_branch_and_tree('from', format='2a')
    to_tree = self.make_branch_and_tree('to', format='development-subtree')
    from_tree.commit(message='first commit')
    out, err = self.run_bzr(['pull', '-d', 'to', 'from'])
    self.assertContainsRe(err, '(?m)Fetching into experimental format')