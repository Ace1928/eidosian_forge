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
def test_worktreeconfig_extension(self):
    r = self._repo
    c = r.get_config()
    c.set(('core',), 'repositoryformatversion', '1')
    c.set(('extensions',), 'worktreeconfig', True)
    c.write_to_path()
    c = r.get_worktree_config()
    c.set(('user',), 'repositoryformatversion', '1')
    c.set((b'user',), b'name', b'Jelmer')
    c.write_to_path()
    cs = r.get_config_stack()
    self.assertEqual(cs.get(('user',), 'name'), b'Jelmer')