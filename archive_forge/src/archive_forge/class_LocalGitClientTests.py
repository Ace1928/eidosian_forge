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
class LocalGitClientTests(TestCase):

    def test_get_url(self):
        path = '/tmp/repo.git'
        c = LocalGitClient()
        url = c.get_url(path)
        self.assertEqual('file:///tmp/repo.git', url)

    def test_fetch_into_empty(self):
        c = LocalGitClient()
        target = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, target)
        t = Repo.init_bare(target)
        self.addCleanup(t.close)
        s = open_repo('a.git')
        self.addCleanup(tear_down_repo, s)
        self.assertEqual(s.get_refs(), c.fetch(s.path, t).refs)

    def test_clone(self):
        c = LocalGitClient()
        s = open_repo('a.git')
        self.addCleanup(tear_down_repo, s)
        target = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, target)
        result_repo = c.clone(s.path, target, mkdir=False)
        self.addCleanup(result_repo.close)
        expected = dict(s.get_refs())
        expected[b'refs/remotes/origin/HEAD'] = expected[b'HEAD']
        expected[b'refs/remotes/origin/master'] = expected[b'refs/heads/master']
        self.assertEqual(expected, result_repo.get_refs())

    def test_fetch_empty(self):
        c = LocalGitClient()
        s = open_repo('a.git')
        self.addCleanup(tear_down_repo, s)
        out = BytesIO()
        walker = {}
        ret = c.fetch_pack(s.path, lambda heads, **kwargs: [], graph_walker=walker, pack_data=out.write)
        self.assertEqual({b'HEAD': b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', b'refs/heads/master': b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', b'refs/tags/mytag': b'28237f4dc30d0d462658d6b937b08a0f0b6ef55a', b'refs/tags/mytag-packed': b'b0931cadc54336e78a1d980420e3268903b57a50'}, ret.refs)
        self.assertEqual({b'HEAD': b'refs/heads/master'}, ret.symrefs)
        self.assertEqual(b'PACK\x00\x00\x00\x02\x00\x00\x00\x00\x02\x9d\x08\x82;\xd8\xa8\xea\xb5\x10\xadj\xc7\\\x82<\xfd>\xd3\x1e', out.getvalue())

    def test_fetch_pack_none(self):
        c = LocalGitClient()
        s = open_repo('a.git')
        self.addCleanup(tear_down_repo, s)
        out = BytesIO()
        walker = MemoryRepo().get_graph_walker()
        ret = c.fetch_pack(s.path, lambda heads, **kwargs: [b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'], graph_walker=walker, pack_data=out.write)
        self.assertEqual({b'HEAD': b'refs/heads/master'}, ret.symrefs)
        self.assertEqual({b'HEAD': b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', b'refs/heads/master': b'a90fa2d900a17e99b433217e988c4eb4a2e9a097', b'refs/tags/mytag': b'28237f4dc30d0d462658d6b937b08a0f0b6ef55a', b'refs/tags/mytag-packed': b'b0931cadc54336e78a1d980420e3268903b57a50'}, ret.refs)
        self.assertTrue(out.getvalue().startswith(b'PACK\x00\x00\x00\x02\x00\x00\x00\x07'))

    def test_send_pack_without_changes(self):
        local = open_repo('a.git')
        self.addCleanup(tear_down_repo, local)
        target = open_repo('a.git')
        self.addCleanup(tear_down_repo, target)
        self.send_and_verify(b'master', local, target)

    def test_send_pack_with_changes(self):
        local = open_repo('a.git')
        self.addCleanup(tear_down_repo, local)
        target_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, target_path)
        with Repo.init_bare(target_path) as target:
            self.send_and_verify(b'master', local, target)

    def test_get_refs(self):
        local = open_repo('refs.git')
        self.addCleanup(tear_down_repo, local)
        client = LocalGitClient()
        refs = client.get_refs(local.path)
        self.assertDictEqual(local.refs.as_dict(), refs)

    def send_and_verify(self, branch, local, target):
        """Send branch from local to remote repository and verify it worked."""
        client = LocalGitClient()
        ref_name = b'refs/heads/' + branch
        result = client.send_pack(target.path, lambda _: {ref_name: local.refs[ref_name]}, local.generate_pack_data)
        self.assertEqual(local.refs[ref_name], result.refs[ref_name])
        self.assertIs(None, result.agent)
        self.assertEqual({}, result.ref_status)
        obj_local = local.get_object(result.refs[ref_name])
        obj_target = target.get_object(result.refs[ref_name])
        self.assertEqual(obj_local, obj_target)