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
def test_send_pack_new_ref(self):
    self.rin.write(b'0064310ca9477129b8586fa2afc779c1f57cf64bba6c refs/heads/master\x00 report-status delete-refs ofs-delta\n0000000eunpack ok\n0019ok refs/heads/blah12\n0000')
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
        return {b'refs/heads/blah12': commit.id, b'refs/heads/master': b'310ca9477129b8586fa2afc779c1f57cf64bba6c'}

    def generate_pack_data(have, want, ofs_delta=False, progress=None):
        return pack_objects_to_data([(commit, None), (tree, b'')])
    f = BytesIO()
    count, records = generate_pack_data(None, None)
    write_pack_data(f.write, records, num_records=count)
    self.client.send_pack(b'/', update_refs, generate_pack_data)
    self.assertEqual(self.rout.getvalue(), b'008b0000000000000000000000000000000000000000 ' + commit.id + b' refs/heads/blah12\x00delete-refs ofs-delta report-status0000' + f.getvalue())