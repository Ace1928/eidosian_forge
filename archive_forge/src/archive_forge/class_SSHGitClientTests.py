import base64
import os
import shutil
import sys
import tempfile
import warnings
from io import BytesIO
from typing import Dict
from unittest.mock import patch
from urllib.parse import quote as urlquote
from urllib.parse import urlparse
import dulwich
from dulwich import client
from dulwich.tests import TestCase, skipIf
from ..client import (
from ..config import ConfigDict
from ..objects import Commit, Tree
from ..pack import pack_objects_to_data, write_pack_data, write_pack_objects
from ..protocol import TCP_GIT_PORT, Protocol
from ..repo import MemoryRepo, Repo
from .utils import open_repo, setup_warning_catcher, tear_down_repo
class SSHGitClientTests(TestCase):

    def setUp(self):
        super().setUp()
        self.server = TestSSHVendor()
        self.real_vendor = client.get_ssh_vendor
        client.get_ssh_vendor = lambda: self.server
        self.client = SSHGitClient('git.samba.org')

    def tearDown(self):
        super().tearDown()
        client.get_ssh_vendor = self.real_vendor

    def test_get_url(self):
        path = '/tmp/repo.git'
        c = SSHGitClient('git.samba.org')
        url = c.get_url(path)
        self.assertEqual('ssh://git.samba.org/tmp/repo.git', url)

    def test_get_url_with_username_and_port(self):
        path = '/tmp/repo.git'
        c = SSHGitClient('git.samba.org', port=2222, username='user')
        url = c.get_url(path)
        self.assertEqual('ssh://user@git.samba.org:2222/tmp/repo.git', url)

    def test_default_command(self):
        self.assertEqual(b'git-upload-pack', self.client._get_cmd_path(b'upload-pack'))

    def test_alternative_command_path(self):
        self.client.alternative_paths[b'upload-pack'] = b'/usr/lib/git/git-upload-pack'
        self.assertEqual(b'/usr/lib/git/git-upload-pack', self.client._get_cmd_path(b'upload-pack'))

    def test_alternative_command_path_spaces(self):
        self.client.alternative_paths[b'upload-pack'] = b'/usr/lib/git/git-upload-pack -ibla'
        self.assertEqual(b'/usr/lib/git/git-upload-pack -ibla', self.client._get_cmd_path(b'upload-pack'))

    def test_connect(self):
        server = self.server
        client = self.client
        client.username = b'username'
        client.port = 1337
        client._connect(b'command', b'/path/to/repo')
        self.assertEqual(b'username', server.username)
        self.assertEqual(1337, server.port)
        self.assertEqual("git-command '/path/to/repo'", server.command)
        client._connect(b'relative-command', b'/~/path/to/repo')
        self.assertEqual("git-relative-command '~/path/to/repo'", server.command)

    def test_ssh_command_precedence(self):
        self.overrideEnv('GIT_SSH', '/path/to/ssh')
        test_client = SSHGitClient('git.samba.org')
        self.assertEqual(test_client.ssh_command, '/path/to/ssh')
        self.overrideEnv('GIT_SSH_COMMAND', '/path/to/ssh -o Option=Value')
        test_client = SSHGitClient('git.samba.org')
        self.assertEqual(test_client.ssh_command, '/path/to/ssh -o Option=Value')
        test_client = SSHGitClient('git.samba.org', ssh_command='ssh -o Option1=Value1')
        self.assertEqual(test_client.ssh_command, 'ssh -o Option1=Value1')