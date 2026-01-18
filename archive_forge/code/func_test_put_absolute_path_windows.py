import os
import sys
import tempfile
from unittest.mock import Mock, MagicMock, call, patch
from libcloud import _init_once
from libcloud.test import LibcloudTestCase, unittest
from libcloud.utils.py3 import StringIO, u, assertRaisesRegex
from libcloud.compute.ssh import ParamikoSSHClient, ShellOutSSHClient, have_paramiko
def test_put_absolute_path_windows(self):
    conn_params = {'hostname': 'dummy.host.org', 'username': 'ubuntu'}
    client = ParamikoSSHClient(**conn_params)
    mock_client = Mock()
    mock_sftp_client = Mock()
    mock_transport = Mock()
    mock_client.get_transport.return_value = mock_transport
    mock_sftp_client.getcwd.return_value = 'C:\\Administrator'
    client.client = mock_client
    client.sftp_client = mock_sftp_client
    result = client.put(path='C:\\users\\user1\\1.txt', contents='foo bar', chmod=455, mode='w')
    self.assertEqual(result, 'C:\\users\\user1\\1.txt')
    result = client.put(path='\\users\\user1\\1.txt', contents='foo bar', chmod=455, mode='w')
    self.assertEqual(result, '\\users\\user1\\1.txt')
    result = client.put(path='1.txt', contents='foo bar', chmod=455, mode='w')
    self.assertEqual(result, 'C:\\Administrator\\1.txt')
    mock_client.get_transport.return_value = mock_transport
    mock_sftp_client.getcwd.return_value = '/C:\\User1'
    client.client = mock_client
    client.sftp_client = mock_sftp_client
    result = client.put(path='1.txt', contents='foo bar', chmod=455, mode='w')
    self.assertEqual(result, 'C:\\User1\\1.txt')