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
class PLinkSSHVendorTests(TestCase):

    def setUp(self):
        self._orig_popen = dulwich.client.subprocess.Popen
        dulwich.client.subprocess.Popen = DummyPopen

    def tearDown(self):
        dulwich.client.subprocess.Popen = self._orig_popen

    def test_run_command_dashes(self):
        vendor = PLinkSSHVendor()
        self.assertRaises(StrangeHostname, vendor.run_command, '--weird-host', 'git-clone-url')

    def test_run_command_password_and_privkey(self):
        vendor = PLinkSSHVendor()
        warnings.simplefilter('always', UserWarning)
        self.addCleanup(warnings.resetwarnings)
        warnings_list, restore_warnings = setup_warning_catcher()
        self.addCleanup(restore_warnings)
        command = vendor.run_command('host', 'git-clone-url', password='12345', key_filename='/tmp/id_rsa')
        expected_warning = UserWarning('Invoking PLink with a password exposes the password in the process list.')
        for w in warnings_list:
            if type(w) is type(expected_warning) and w.args == expected_warning.args:
                break
        else:
            raise AssertionError(f'Expected warning {expected_warning!r} not in {warnings_list!r}')
        args = command.proc.args
        if sys.platform == 'win32':
            binary = ['plink.exe', '-ssh']
        else:
            binary = ['plink', '-ssh']
        expected = [*binary, '-pw', '12345', '-i', '/tmp/id_rsa', 'host', 'git-clone-url']
        self.assertListEqual(expected, args[0])

    def test_run_command_password(self):
        if sys.platform == 'win32':
            binary = ['plink.exe', '-ssh']
        else:
            binary = ['plink', '-ssh']
        expected = [*binary, '-pw', '12345', 'host', 'git-clone-url']
        vendor = PLinkSSHVendor()
        warnings.simplefilter('always', UserWarning)
        self.addCleanup(warnings.resetwarnings)
        warnings_list, restore_warnings = setup_warning_catcher()
        self.addCleanup(restore_warnings)
        command = vendor.run_command('host', 'git-clone-url', password='12345')
        expected_warning = UserWarning('Invoking PLink with a password exposes the password in the process list.')
        for w in warnings_list:
            if type(w) is type(expected_warning) and w.args == expected_warning.args:
                break
        else:
            raise AssertionError(f'Expected warning {expected_warning!r} not in {warnings_list!r}')
        args = command.proc.args
        self.assertListEqual(expected, args[0])

    def test_run_command_with_port_username_and_privkey(self):
        if sys.platform == 'win32':
            binary = ['plink.exe', '-ssh']
        else:
            binary = ['plink', '-ssh']
        expected = [*binary, '-P', '2200', '-i', '/tmp/id_rsa', 'user@host', 'git-clone-url']
        vendor = PLinkSSHVendor()
        command = vendor.run_command('host', 'git-clone-url', username='user', port='2200', key_filename='/tmp/id_rsa')
        args = command.proc.args
        self.assertListEqual(expected, args[0])

    def test_run_with_ssh_command(self):
        expected = ['/path/to/plink', '-x', 'host', 'git-clone-url']
        vendor = SubprocessSSHVendor()
        command = vendor.run_command('host', 'git-clone-url', ssh_command='/path/to/plink')
        args = command.proc.args
        self.assertListEqual(expected, args[0])