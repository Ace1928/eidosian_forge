import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
def test_put_absolute_path(self):
    conn_params = {'hostname': 'dummy.host.org', 'username': 'ubuntu'}
    client = ParamikoSSHClient(**conn_params)
    mock_client = Mock()
    mock_sftp_client = Mock()
    mock_transport = Mock()
    mock_client.get_transport.return_value = mock_transport
    mock_sftp_client.getcwd.return_value = '/mock/cwd'
    client.client = mock_client
    client.sftp_client = mock_sftp_client
    result = client.put(path='/test/remote/path.txt', contents='foo bar', chmod=455, mode='w')
    self.assertEqual(result, '/test/remote/path.txt')
    calls = [call('/'), call('test'), call('remote')]
    mock_sftp_client.chdir.assert_has_calls(calls, any_order=False)
    calls = [call('path.txt', mode='w'), call().write('foo bar'), call().chmod(455), call().close()]
    mock_sftp_client.file.assert_has_calls(calls, any_order=False)