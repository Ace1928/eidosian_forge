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