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
class TCPGitClientTests(TestCase):

    def test_get_url(self):
        host = 'github.com'
        path = '/jelmer/dulwich'
        c = TCPGitClient(host)
        url = c.get_url(path)
        self.assertEqual('git://github.com/jelmer/dulwich', url)

    def test_get_url_with_port(self):
        host = 'github.com'
        path = '/jelmer/dulwich'
        port = 9090
        c = TCPGitClient(host, port=port)
        url = c.get_url(path)
        self.assertEqual('git://github.com:9090/jelmer/dulwich', url)