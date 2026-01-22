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
class ServeCommandTests(TestCase):
    """Tests for serve_command."""

    def setUp(self):
        super().setUp()
        self.backend = DictBackend({})

    def serve_command(self, handler_cls, args, inf, outf):
        return serve_command(handler_cls, [b'test', *args], backend=self.backend, inf=inf, outf=outf)

    def test_receive_pack(self):
        commit = make_commit(id=ONE, parents=[], commit_time=111)
        self.backend.repos[b'/'] = MemoryRepo.init_bare([commit], {b'refs/heads/master': commit.id})
        outf = BytesIO()
        exitcode = self.serve_command(ReceivePackHandler, [b'/'], BytesIO(b'0000'), outf)
        outlines = outf.getvalue().splitlines()
        self.assertEqual(2, len(outlines))
        self.assertEqual(b'1111111111111111111111111111111111111111 refs/heads/master', outlines[0][4:].split(b'\x00')[0])
        self.assertEqual(b'0000', outlines[-1])
        self.assertEqual(0, exitcode)