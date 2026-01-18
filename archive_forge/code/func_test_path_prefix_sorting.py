import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
def test_path_prefix_sorting(self):
    """Doing a sort on path prefix should match our sample data."""
    original_paths = ['a', 'a/b', 'a/b/c', 'b', 'b/c', 'd', 'd/e', 'd/e/f', 'd/f', 'd/g', 'g']
    dir_sorted_paths = ['a', 'b', 'd', 'g', 'a/b', 'a/b/c', 'b/c', 'd/e', 'd/f', 'd/g', 'd/e/f']
    self.assertEqual(dir_sorted_paths, sorted(original_paths, key=osutils.path_prefix_key))
    self.assertEqual(dir_sorted_paths, sorted(original_paths, key=osutils.path_prefix_key))