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
def test_run_with_ssh_command(self):
    expected = ['/path/to/plink', '-x', 'host', 'git-clone-url']
    vendor = SubprocessSSHVendor()
    command = vendor.run_command('host', 'git-clone-url', ssh_command='/path/to/plink')
    args = command.proc.args
    self.assertListEqual(expected, args[0])