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
def test_update_shallow(self):
    self._repo.update_shallow(None, None)
    self.assertEqual(set(), self._repo.get_shallow())
    self._repo.update_shallow([b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'], None)
    self.assertEqual({b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'}, self._repo.get_shallow())
    self._repo.update_shallow([b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'], [b'f9e39b120c68182a4ba35349f832d0e4e61f485c'])
    self.assertEqual({b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'}, self._repo.get_shallow())
    self._repo.update_shallow(None, [b'a90fa2d900a17e99b433217e988c4eb4a2e9a097'])
    self.assertEqual(set(), self._repo.get_shallow())
    self.assertEqual(False, os.path.exists(os.path.join(self._repo.controldir(), 'shallow')))