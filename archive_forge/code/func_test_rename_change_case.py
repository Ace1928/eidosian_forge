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
def test_rename_change_case(self):
    self.build_tree(['a', 'b/'])
    osutils.rename('a', 'A')
    osutils.rename('b', 'B')
    shape = sorted(os.listdir('.'))
    self.assertEqual(['A', 'B'], shape)