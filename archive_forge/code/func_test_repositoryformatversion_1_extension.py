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
def test_repositoryformatversion_1_extension(self):
    r = self._repo
    c = r.get_config()
    c.set(('core',), 'repositoryformatversion', '1')
    c.set(('extensions',), 'unknownextension', True)
    c.write_to_path()
    self.assertRaises(UnsupportedExtension, Repo, self._repo_dir)