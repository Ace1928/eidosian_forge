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
def test_send_pack_no_sideband64k_with_update_ref_error(self) -> None:
    pkts = [b'55dcc6bf963f922e1ed5c4bbaaefcfacef57b1d7 capabilities^{}\x00 report-status delete-refs ofs-delta\n', b'', b'unpack ok', b'ng refs/foo/bar pre-receive hook declined', b'']
    for pkt in pkts:
        if pkt == b'':
            self.rin.write(b'0000')
        else:
            self.rin.write(('%04x' % (len(pkt) + 4)).encode('ascii') + pkt)
    self.rin.seek(0)
    tree = Tree()
    commit = Commit()
    commit.tree = tree
    commit.parents = []
    commit.author = commit.committer = b'test user'
    commit.commit_time = commit.author_time = 1174773719
    commit.commit_timezone = commit.author_timezone = 0
    commit.encoding = b'UTF-8'
    commit.message = b'test message'

    def update_refs(refs):
        return {b'refs/foo/bar': commit.id}

    def generate_pack_data(have, want, ofs_delta=False, progress=None):
        return pack_objects_to_data([(commit, None), (tree, b'')])
    result = self.client.send_pack('blah', update_refs, generate_pack_data)
    self.assertEqual({b'refs/foo/bar': 'pre-receive hook declined'}, result.ref_status)
    self.assertEqual({b'refs/foo/bar': commit.id}, result.refs)