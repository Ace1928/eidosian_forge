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
def test_status(self):
    os.mkdir('branch1')
    os.chdir('branch1')
    self.run_bzr('init')
    self.run_bzr('commit --unchanged --message f')
    self.run_bzr('branch . ../branch2')
    self.run_bzr('branch . ../branch3')
    self.run_bzr('commit --unchanged --message peter')
    os.chdir('../branch2')
    self.run_bzr('merge ../branch1')
    self.run_bzr('commit --unchanged --message pumpkin')
    os.chdir('../branch3')
    self.run_bzr('merge ../branch2')
    message = self.run_bzr('status')[0]