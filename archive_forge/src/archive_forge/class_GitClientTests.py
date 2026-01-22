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
class GitClientTests(TestCase):

    def setUp(self):
        super().setUp()
        self.rout = BytesIO()
        self.rin = BytesIO()
        self.client = DummyClient(lambda x: True, self.rin.read, self.rout.write)

    def test_caps(self):
        agent_cap = ('agent=dulwich/%d.%d.%d' % dulwich.__version__).encode('ascii')
        self.assertEqual({b'multi_ack', b'side-band-64k', b'ofs-delta', b'thin-pack', b'multi_ack_detailed', b'shallow', agent_cap}, set(self.client._fetch_capabilities))
        self.assertEqual({b'delete-refs', b'ofs-delta', b'report-status', b'side-band-64k', agent_cap}, set(self.client._send_capabilities))

    def test_archive_ack(self):
        self.rin.write(b'0009NACK\n0000')
        self.rin.seek(0)
        self.client.archive(b'bla', b'HEAD', None, None)
        self.assertEqual(self.rout.getvalue(), b'0011argument HEAD0000')

    def test_fetch_empty(self):
        self.rin.write(b'0000')
        self.rin.seek(0)

        def check_heads(heads, **kwargs):
            self.assertEqual(heads, {})
            return []
        ret = self.client.fetch_pack(b'/', check_heads, None, None)
        self.assertEqual({}, ret.refs)
        self.assertEqual({}, ret.symrefs)

    def test_fetch_pack_ignores_magic_ref(self):
        self.rin.write(b'00000000000000000000000000000000000000000000 capabilities^{}\x00 multi_ack thin-pack side-band side-band-64k ofs-delta shallow no-progress include-tag\n0000')
        self.rin.seek(0)

        def check_heads(heads, **kwargs):
            self.assertEqual({}, heads)
            return []
        ret = self.client.fetch_pack(b'bla', check_heads, None, None, None)
        self.assertEqual({}, ret.refs)
        self.assertEqual({}, ret.symrefs)
        self.assertEqual(self.rout.getvalue(), b'0000')

    def test_fetch_pack_none(self):
        self.rin.write(b'008855dcc6bf963f922e1ed5c4bbaaefcfacef57b1d7 HEAD\x00multi_ack thin-pack side-band side-band-64k ofs-delta shallow no-progress include-tag\n0000')
        self.rin.seek(0)
        ret = self.client.fetch_pack(b'bla', lambda heads, **kwargs: [], None, None, None)
        self.assertEqual({b'HEAD': b'55dcc6bf963f922e1ed5c4bbaaefcfacef57b1d7'}, ret.refs)
        self.assertEqual({}, ret.symrefs)
        self.assertEqual(self.rout.getvalue(), b'0000')

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

    def test_send_pack_none(self):
        self.rin.write(b'0078310ca9477129b8586fa2afc779c1f57cf64bba6c refs/heads/master\x00 report-status delete-refs side-band-64k quiet ofs-delta\n0000')
        self.rin.seek(0)

        def update_refs(refs):
            return {b'refs/heads/master': b'310ca9477129b8586fa2afc779c1f57cf64bba6c'}

        def generate_pack_data(have, want, ofs_delta=False, progress=None):
            return (0, [])
        self.client.send_pack(b'/', update_refs, generate_pack_data)
        self.assertEqual(self.rout.getvalue(), b'0000')

    def test_send_pack_keep_and_delete(self):
        self.rin.write(b'0063310ca9477129b8586fa2afc779c1f57cf64bba6c refs/heads/master\x00report-status delete-refs ofs-delta\n003f310ca9477129b8586fa2afc779c1f57cf64bba6c refs/heads/keepme\n0000000eunpack ok\n0019ok refs/heads/master\n0000')
        self.rin.seek(0)

        def update_refs(refs):
            return {b'refs/heads/master': b'0' * 40}

        def generate_pack_data(have, want, ofs_delta=False, progress=None):
            return (0, [])
        self.client.send_pack(b'/', update_refs, generate_pack_data)
        self.assertEqual(self.rout.getvalue(), b'008b310ca9477129b8586fa2afc779c1f57cf64bba6c 0000000000000000000000000000000000000000 refs/heads/master\x00delete-refs ofs-delta report-status0000')

    def test_send_pack_delete_only(self):
        self.rin.write(b'0063310ca9477129b8586fa2afc779c1f57cf64bba6c refs/heads/master\x00report-status delete-refs ofs-delta\n0000000eunpack ok\n0019ok refs/heads/master\n0000')
        self.rin.seek(0)

        def update_refs(refs):
            return {b'refs/heads/master': b'0' * 40}

        def generate_pack_data(have, want, ofs_delta=False, progress=None):
            return (0, [])
        self.client.send_pack(b'/', update_refs, generate_pack_data)
        self.assertEqual(self.rout.getvalue(), b'008b310ca9477129b8586fa2afc779c1f57cf64bba6c 0000000000000000000000000000000000000000 refs/heads/master\x00delete-refs ofs-delta report-status0000')

    def test_send_pack_new_ref_only(self):
        self.rin.write(b'0063310ca9477129b8586fa2afc779c1f57cf64bba6c refs/heads/master\x00report-status delete-refs ofs-delta\n0000000eunpack ok\n0019ok refs/heads/blah12\n0000')
        self.rin.seek(0)

        def update_refs(refs):
            return {b'refs/heads/blah12': b'310ca9477129b8586fa2afc779c1f57cf64bba6c', b'refs/heads/master': b'310ca9477129b8586fa2afc779c1f57cf64bba6c'}

        def generate_pack_data(have, want, ofs_delta=False, progress=None):
            return (0, [])
        f = BytesIO()
        write_pack_objects(f.write, [])
        self.client.send_pack('/', update_refs, generate_pack_data)
        self.assertEqual(self.rout.getvalue(), b'008b0000000000000000000000000000000000000000 310ca9477129b8586fa2afc779c1f57cf64bba6c refs/heads/blah12\x00delete-refs ofs-delta report-status0000' + f.getvalue())

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

    def test_send_pack_no_deleteref_delete_only(self):
        pkts = [b'310ca9477129b8586fa2afc779c1f57cf64bba6c refs/heads/master\x00 report-status ofs-delta\n', b'', b'']
        for pkt in pkts:
            if pkt == b'':
                self.rin.write(b'0000')
            else:
                self.rin.write(('%04x' % (len(pkt) + 4)).encode('ascii') + pkt)
        self.rin.seek(0)

        def update_refs(refs):
            return {b'refs/heads/master': b'0' * 40}

        def generate_pack_data(have, want, ofs_delta=False, progress=None):
            return (0, [])
        result = self.client.send_pack(b'/', update_refs, generate_pack_data)
        self.assertEqual(result.ref_status, {b'refs/heads/master': 'remote does not support deleting refs'})
        self.assertEqual(result.refs, {b'refs/heads/master': b'310ca9477129b8586fa2afc779c1f57cf64bba6c'})
        self.assertEqual(self.rout.getvalue(), b'0000')