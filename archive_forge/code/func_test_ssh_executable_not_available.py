import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
def test_ssh_executable_not_available(self):

    class MockChild:
        returncode = 127

        def communicate(*args, **kwargs):
            pass

    def mock_popen(*args, **kwargs):
        return MockChild()
    with patch('subprocess.Popen', mock_popen):
        try:
            ShellOutSSHClient(hostname='localhost', username='foo')
        except ValueError as e:
            msg = str(e)
            self.assertTrue('ssh client is not available' in msg)
        else:
            self.fail('Exception was not thrown')