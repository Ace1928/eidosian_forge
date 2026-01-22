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
def assertFilesystemHidden(self, path):
    if sys.platform != 'win32':
        return
    import ctypes
    from ctypes.wintypes import DWORD, LPCWSTR
    GetFileAttributesW = ctypes.WINFUNCTYPE(DWORD, LPCWSTR)(('GetFileAttributesW', ctypes.windll.kernel32))
    self.assertTrue(2 & GetFileAttributesW(path))