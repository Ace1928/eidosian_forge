import os
import shutil
import sys
import tempfile
from io import BytesIO
from typing import Dict, List
from dulwich.tests import TestCase
from ..errors import (
from ..object_store import MemoryObjectStore
from ..objects import Tree
from ..protocol import ZERO_SHA, format_capability_line
from ..repo import MemoryRepo, Repo
from ..server import (
from .utils import make_commit, make_tag
class ProtocolGraphWalkerTestCase(TestCase):

    def setUp(self):
        super().setUp()
        commits = [make_commit(id=ONE, parents=[], commit_time=111), make_commit(id=TWO, parents=[ONE], commit_time=222), make_commit(id=THREE, parents=[ONE], commit_time=333), make_commit(id=FOUR, parents=[TWO], commit_time=444), make_commit(id=FIVE, parents=[THREE], commit_time=555)]
        self._repo = MemoryRepo.init_bare(commits, {})
        backend = DictBackend({b'/': self._repo})
        self._walker = _ProtocolGraphWalker(TestUploadPackHandler(backend, [b'/', b'host=lolcats'], TestProto()), self._repo.object_store, self._repo.get_peeled, self._repo.refs.get_symrefs)

    def test_all_wants_satisfied_no_haves(self):
        self._walker.set_wants([ONE])
        self.assertFalse(self._walker.all_wants_satisfied([]))
        self._walker.set_wants([TWO])
        self.assertFalse(self._walker.all_wants_satisfied([]))
        self._walker.set_wants([THREE])
        self.assertFalse(self._walker.all_wants_satisfied([]))

    def test_all_wants_satisfied_have_root(self):
        self._walker.set_wants([ONE])
        self.assertTrue(self._walker.all_wants_satisfied([ONE]))
        self._walker.set_wants([TWO])
        self.assertTrue(self._walker.all_wants_satisfied([ONE]))
        self._walker.set_wants([THREE])
        self.assertTrue(self._walker.all_wants_satisfied([ONE]))

    def test_all_wants_satisfied_have_branch(self):
        self._walker.set_wants([TWO])
        self.assertTrue(self._walker.all_wants_satisfied([TWO]))
        self._walker.set_wants([THREE])
        self.assertFalse(self._walker.all_wants_satisfied([TWO]))

    def test_all_wants_satisfied(self):
        self._walker.set_wants([FOUR, FIVE])
        self.assertTrue(self._walker.all_wants_satisfied([FOUR, FIVE]))
        self.assertTrue(self._walker.all_wants_satisfied([ONE]))
        self.assertFalse(self._walker.all_wants_satisfied([TWO]))
        self.assertFalse(self._walker.all_wants_satisfied([THREE]))
        self.assertTrue(self._walker.all_wants_satisfied([TWO, THREE]))

    def test_split_proto_line(self):
        allowed = (b'want', b'done', None)
        self.assertEqual((b'want', ONE), _split_proto_line(b'want ' + ONE + b'\n', allowed))
        self.assertEqual((b'want', TWO), _split_proto_line(b'want ' + TWO + b'\n', allowed))
        self.assertRaises(GitProtocolError, _split_proto_line, b'want xxxx\n', allowed)
        self.assertRaises(UnexpectedCommandError, _split_proto_line, b'have ' + THREE + b'\n', allowed)
        self.assertRaises(GitProtocolError, _split_proto_line, b'foo ' + FOUR + b'\n', allowed)
        self.assertRaises(GitProtocolError, _split_proto_line, b'bar', allowed)
        self.assertEqual((b'done', None), _split_proto_line(b'done\n', allowed))
        self.assertEqual((None, None), _split_proto_line(b'', allowed))

    def test_determine_wants(self):
        self._walker.proto.set_output([None])
        self.assertEqual([], self._walker.determine_wants({}))
        self.assertEqual(None, self._walker.proto.get_received_line())
        self._walker.proto.set_output([b'want ' + ONE + b' multi_ack', b'want ' + TWO, None])
        heads = {b'refs/heads/ref1': ONE, b'refs/heads/ref2': TWO, b'refs/heads/ref3': THREE}
        self._repo.refs._update(heads)
        self.assertEqual([ONE, TWO], self._walker.determine_wants(heads))
        self._walker.advertise_refs = True
        self.assertEqual([], self._walker.determine_wants(heads))
        self._walker.advertise_refs = False
        self._walker.proto.set_output([b'want ' + FOUR + b' multi_ack', None])
        self.assertRaises(GitProtocolError, self._walker.determine_wants, heads)
        self._walker.proto.set_output([None])
        self.assertEqual([], self._walker.determine_wants(heads))
        self._walker.proto.set_output([b'want ' + ONE + b' multi_ack', b'foo', None])
        self.assertRaises(GitProtocolError, self._walker.determine_wants, heads)
        self._walker.proto.set_output([b'want ' + FOUR + b' multi_ack', None])
        self.assertRaises(GitProtocolError, self._walker.determine_wants, heads)

    def test_determine_wants_advertisement(self):
        self._walker.proto.set_output([None])
        heads = {b'refs/heads/ref4': FOUR, b'refs/heads/ref5': FIVE, b'refs/heads/tag6': SIX}
        self._repo.refs._update(heads)
        self._repo.refs._update_peeled(heads)
        self._repo.refs._update_peeled({b'refs/heads/tag6': FIVE})
        self._walker.determine_wants(heads)
        lines = []
        while True:
            line = self._walker.proto.get_received_line()
            if line is None:
                break
            if b'\x00' in line:
                line = line[:line.index(b'\x00')]
            lines.append(line.rstrip())
        self.assertEqual([FOUR + b' refs/heads/ref4', FIVE + b' refs/heads/ref5', FIVE + b' refs/heads/tag6^{}', SIX + b' refs/heads/tag6'], sorted(lines))
        for i, line in enumerate(lines):
            if line.endswith(b' refs/heads/tag6'):
                self.assertEqual(FIVE + b' refs/heads/tag6^{}', lines[i + 1])

    def _handle_shallow_request(self, lines, heads):
        self._walker.proto.set_output([*lines, None])
        self._walker._handle_shallow_request(heads)

    def assertReceived(self, expected):
        self.assertEqual(expected, list(iter(self._walker.proto.get_received_line, None)))

    def test_handle_shallow_request_no_client_shallows(self):
        self._handle_shallow_request([b'deepen 2\n'], [FOUR, FIVE])
        self.assertEqual({TWO, THREE}, self._walker.shallow)
        self.assertReceived([b'shallow ' + TWO, b'shallow ' + THREE])

    def test_handle_shallow_request_no_new_shallows(self):
        lines = [b'shallow ' + TWO + b'\n', b'shallow ' + THREE + b'\n', b'deepen 2\n']
        self._handle_shallow_request(lines, [FOUR, FIVE])
        self.assertEqual({TWO, THREE}, self._walker.shallow)
        self.assertReceived([])

    def test_handle_shallow_request_unshallows(self):
        lines = [b'shallow ' + TWO + b'\n', b'deepen 3\n']
        self._handle_shallow_request(lines, [FOUR, FIVE])
        self.assertEqual({ONE}, self._walker.shallow)
        self.assertReceived([b'shallow ' + ONE, b'unshallow ' + TWO])