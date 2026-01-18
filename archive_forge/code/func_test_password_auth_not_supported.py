import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
def test_password_auth_not_supported(self):
    try:
        ShellOutSSHClient(hostname='localhost', username='foo', password='bar')
    except ValueError as e:
        msg = str(e)
        self.assertTrue('ShellOutSSHClient only supports key auth' in msg)
    else:
        self.fail('Exception was not thrown')