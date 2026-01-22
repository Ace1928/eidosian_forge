import os
import re
import sys
import breezy
from breezy import osutils
from breezy.branch import Branch
from breezy.errors import CommandError
from breezy.tests import TestCaseWithTransport
from breezy.tests.http_utils import TestCaseWithWebserver
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
class RemoteTests:
    """Test brz ui commands against remote branches."""

    def test_branch(self):
        os.mkdir('from')
        wt = self.make_branch_and_tree('from')
        branch = wt.branch
        wt.commit('empty commit for nonsense', allow_pointless=True)
        url = self.get_readonly_url('from')
        self.run_bzr(['branch', url, 'to'])
        branch = Branch.open('to')
        self.assertEqual(1, branch.last_revision_info()[0])
        self.assertEqual(url + '/', branch.get_parent())

    def test_log(self):
        self.build_tree(['branch/', 'branch/file'])
        self.run_bzr('init branch')[0]
        self.run_bzr('add branch/file')[0]
        self.run_bzr('commit -m foo branch')[0]
        url = self.get_readonly_url('branch/file')
        output = self.run_bzr('log %s' % url)[0]
        self.assertEqual(8, len(output.split('\n')))

    def test_check(self):
        self.build_tree(['branch/', 'branch/file'])
        self.run_bzr('init branch')[0]
        self.run_bzr('add branch/file')[0]
        self.run_bzr('commit -m foo branch')[0]
        url = self.get_readonly_url('branch/')
        self.run_bzr(['check', url])

    def test_push(self):
        os.mkdir('my-branch')
        os.chdir('my-branch')
        self.run_bzr('init')
        with open('hello', 'w') as f:
            f.write('foo')
        self.run_bzr('add hello')
        self.run_bzr('commit -m setup')
        self.run_bzr(['push', self.get_url('output-branch')])