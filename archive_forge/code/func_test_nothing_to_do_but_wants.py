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
def test_nothing_to_do_but_wants(self):
    refs = {b'refs/tags/tag1': ONE}
    tree = Tree()
    self._repo.object_store.add_object(tree)
    self._repo.object_store.add_object(make_commit(id=ONE, tree=tree))
    for name, sha in refs.items():
        self._repo.refs[name] = sha
    self._handler.proto.set_output([b'want ' + ONE + b' side-band-64k thin-pack ofs-delta', None, b'have ' + ONE, b'done', None])
    self._handler.handle()
    self.assertTrue(self._handler.proto.get_received_line(1).startswith(b'PACK'))