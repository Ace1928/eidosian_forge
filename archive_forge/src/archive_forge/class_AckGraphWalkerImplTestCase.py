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
class AckGraphWalkerImplTestCase(TestCase):
    """Base setup and asserts for AckGraphWalker tests."""

    def setUp(self):
        super().setUp()
        self._walker = TestProtocolGraphWalker()
        self._walker.lines = [(b'have', TWO), (b'have', ONE), (b'have', THREE), (b'done', None)]
        self._impl = self.impl_cls(self._walker)
        self._walker._impl = self._impl

    def assertNoAck(self):
        self.assertEqual(None, self._walker.pop_ack())

    def assertAcks(self, acks):
        for sha, ack_type in acks:
            self.assertEqual((sha, ack_type), self._walker.pop_ack())
        self.assertNoAck()

    def assertAck(self, sha, ack_type=b''):
        self.assertAcks([(sha, ack_type)])

    def assertNak(self):
        self.assertAck(None, b'nak')

    def assertNextEquals(self, sha):
        self.assertEqual(sha, next(self._impl))

    def assertNextEmpty(self):
        self.assertRaises(IndexError, next, self._impl)
        self._walker.handle_done()