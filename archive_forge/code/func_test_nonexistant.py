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
def test_nonexistant(self):
    repo = MemoryRepo.init_bare([], {})
    backend = DictBackend({b'/': repo})
    self.assertRaises(NotGitRepository, backend.open_repository, '/does/not/exist/unless/foo')