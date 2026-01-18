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