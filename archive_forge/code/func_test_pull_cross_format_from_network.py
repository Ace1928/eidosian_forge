import os
import sys
from breezy import (branch, debug, osutils, tests, uncommit, urlutils,
from breezy.bzr import remote
from breezy.directory_service import directories
from breezy.tests import fixtures, script
def test_pull_cross_format_from_network(self):
    self.setup_smart_server_with_call_log()
    from_tree = self.make_branch_and_tree('from', format='2a')
    to_tree = self.make_branch_and_tree('to', format='1.14-rich-root')
    self.assertIsInstance(from_tree.branch, remote.RemoteBranch)
    from_tree.commit(message='first commit')
    out, err = self.run_bzr(['pull', '-d', 'to', from_tree.branch.controldir.root_transport.base])
    self.assertContainsRe(err, '(?m)Doing on-the-fly conversion')