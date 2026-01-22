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
class HandlerTestCase(TestCase):

    def setUp(self):
        super().setUp()
        self._handler = TestGenericPackHandler()

    def assertSucceeds(self, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except GitProtocolError as e:
            self.fail(e)

    def test_capability_line(self):
        self.assertEqual(b' cap1 cap2 cap3', format_capability_line([b'cap1', b'cap2', b'cap3']))

    def test_set_client_capabilities(self):
        set_caps = self._handler.set_client_capabilities
        self.assertSucceeds(set_caps, [b'cap2'])
        self.assertSucceeds(set_caps, [b'cap1', b'cap2'])
        self.assertSucceeds(set_caps, [b'cap3', b'cap1', b'cap2'])
        self.assertRaises(GitProtocolError, set_caps, [b'capxxx', b'cap2'])
        self.assertRaises(GitProtocolError, set_caps, [b'cap1', b'cap3'])
        self.assertRaises(GitProtocolError, set_caps, [b'cap2', b'ignoreme'])
        self.assertNotIn(b'ignoreme', self._handler.capabilities())
        self._handler.innocuous_capabilities = lambda: (b'ignoreme',)
        self.assertSucceeds(set_caps, [b'cap2', b'ignoreme'])

    def test_has_capability(self):
        self.assertRaises(GitProtocolError, self._handler.has_capability, b'cap')
        caps = self._handler.capabilities()
        self._handler.set_client_capabilities(caps)
        for cap in caps:
            self.assertTrue(self._handler.has_capability(cap))
        self.assertFalse(self._handler.has_capability(b'capxxx'))