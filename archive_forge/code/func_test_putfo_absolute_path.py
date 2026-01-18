import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
def test_putfo_absolute_path(self):
    conn_params = {'hostname': 'dummy.host.org', 'username': 'ubuntu'}
    client = ParamikoSSHClient(**conn_params)
    mock_client = Mock()
    mock_sftp_client = Mock()
    mock_transport = Mock()
    mock_client.get_transport.return_value = mock_transport
    mock_sftp_client.getcwd.return_value = '/mock/cwd'
    client.client = mock_client
    client.sftp_client = mock_sftp_client
    mock_fo = StringIO('mock stream data 1')
    result = client.putfo(path='/test/remote/path.txt', fo=mock_fo, chmod=455)
    self.assertEqual(result, '/test/remote/path.txt')
    calls = [call('/'), call('test'), call('remote')]
    mock_sftp_client.chdir.assert_has_calls(calls, any_order=False)
    mock_sftp_client.putfo.assert_called_once_with(mock_fo, '/test/remote/path.txt')
    calls = [call('path.txt'), call().chmod(455), call().close()]
    mock_sftp_client.file.assert_has_calls(calls, any_order=False)