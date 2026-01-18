import glob
import locale
import os
import shutil
import stat
import sys
import tempfile
import warnings
from dulwich import errors, objects, porcelain
from dulwich.tests import TestCase, skipIf
from ..config import Config
from ..errors import NotGitRepository
from ..object_store import tree_lookup_path
from ..repo import (
from .utils import open_repo, setup_warning_catcher, tear_down_repo
import sys
from dulwich.repo import Repo
def test_unstage_remove_file(self):
    file = 'foo'
    full_path = os.path.join(self._repo.path, file)
    with open(full_path, 'w') as f:
        f.write('hello')
    porcelain.add(self._repo, paths=[full_path])
    porcelain.commit(self._repo, message=b'unitest', committer=b'Jane <jane@example.com>', author=b'John <john@example.com>')
    os.remove(full_path)
    self._repo.unstage([file])
    status = list(porcelain.status(self._repo))
    self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [b'foo'], []], status)