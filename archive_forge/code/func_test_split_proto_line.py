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