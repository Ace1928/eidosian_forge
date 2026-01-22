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
class CheckUserIdentityTests(TestCase):

    def test_valid(self):
        check_user_identity(b'Me <me@example.com>')

    def test_invalid(self):
        self.assertRaises(InvalidUserIdentity, check_user_identity, b'No Email')
        self.assertRaises(InvalidUserIdentity, check_user_identity, b'Fullname <missing')
        self.assertRaises(InvalidUserIdentity, check_user_identity, b'Fullname missing>')
        self.assertRaises(InvalidUserIdentity, check_user_identity, b'Fullname >order<>')
        self.assertRaises(InvalidUserIdentity, check_user_identity, b'Contains\x00null byte <>')
        self.assertRaises(InvalidUserIdentity, check_user_identity, b'Contains\nnewline byte <>')