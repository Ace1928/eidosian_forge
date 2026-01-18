import os
import sys
from breezy import (branch, debug, osutils, tests, uncommit, urlutils,
from breezy.bzr import remote
from breezy.directory_service import directories
from breezy.tests import fixtures, script
def test_pull_smart_stacked_streaming_acceptance(self):
    """'brz pull -r 123' works on stacked, smart branches, even when the
        revision specified by the revno is only present in the fallback
        repository.

        See <https://launchpad.net/bugs/380314>
        """
    self.setup_smart_server_with_call_log()
    parent = self.make_branch_and_tree('parent', format='1.9')
    parent.commit(message='first commit')
    parent.commit(message='second commit')
    local = parent.controldir.sprout('local').open_workingtree()
    local.commit(message='local commit')
    local.branch.create_clone_on_transport(self.get_transport('stacked'), stacked_on=self.get_url('parent'))
    empty = self.make_branch_and_tree('empty', format='1.9')
    self.reset_smart_call_log()
    self.run_bzr(['pull', '-r', '1', self.get_url('stacked')], working_dir='empty')
    self.assertLength(20, self.hpss_calls)
    self.assertLength(1, self.hpss_connections)
    remote = branch.Branch.open('stacked')
    self.assertEndsWith(remote.get_stacked_on_url(), '/parent')