import os
import sys
import tempfile
from io import BytesIO
from typing import ClassVar, Dict
from dulwich import errors
from dulwich.tests import SkipTest, TestCase
from ..file import GitFile
from ..objects import ZERO_SHA
from ..refs import (
from ..repo import Repo
from .utils import open_repo, tear_down_repo
class ParseSymrefValueTests(TestCase):

    def test_valid(self):
        self.assertEqual(b'refs/heads/foo', parse_symref_value(b'ref: refs/heads/foo'))

    def test_invalid(self):
        self.assertRaises(ValueError, parse_symref_value, b'foobar')