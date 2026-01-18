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
def test_unstage_midify_file_with_dir(self):
    os.mkdir(os.path.join(self._repo.path, 'new_dir'))
    full_path = os.path.join(self._repo.path, 'new_dir', 'foo')
    with open(full_path, 'w') as f:
        f.write('hello')
    porcelain.add(self._repo, paths=[full_path])
    porcelain.commit(self._repo, message=b'unitest', committer=b'Jane <jane@example.com>', author=b'John <john@example.com>')
    with open(full_path, 'a') as f:
        f.write('something new')
    self._repo.unstage(['new_dir/foo'])
    status = list(porcelain.status(self._repo))
    self.assertEqual([{'add': [], 'delete': [], 'modify': []}, [b'new_dir/foo'], []], status)