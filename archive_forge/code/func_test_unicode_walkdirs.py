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
def test_unicode_walkdirs(self):
    """Walkdirs should always return unicode paths."""
    self.requireFeature(features.UnicodeFilenameFeature)
    name0 = '0file-¶'
    name1 = '1dir-جو'
    name2 = '2file-س'
    tree = [name0, name1 + '/', name1 + '/' + name0, name1 + '/' + name1 + '/', name2]
    self.build_tree(tree)
    expected_dirblocks = [(('', '.'), [(name0, name0, 'file', './' + name0), (name1, name1, 'directory', './' + name1), (name2, name2, 'file', './' + name2)]), ((name1, './' + name1), [(name1 + '/' + name0, name0, 'file', './' + name1 + '/' + name0), (name1 + '/' + name1, name1, 'directory', './' + name1 + '/' + name1)]), ((name1 + '/' + name1, './' + name1 + '/' + name1), [])]
    result = list(osutils.walkdirs('.'))
    self._filter_out_stat(result)
    self.assertEqual(expected_dirblocks, result)
    result = list(osutils.walkdirs('./' + name1, name1))
    self._filter_out_stat(result)
    self.assertEqual(expected_dirblocks[1:], result)