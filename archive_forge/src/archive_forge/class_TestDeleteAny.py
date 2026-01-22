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
class TestDeleteAny(tests.TestCaseInTempDir):

    def test_delete_any_readonly(self):
        self.build_tree(['d/', 'f'])
        osutils.make_readonly('d')
        osutils.make_readonly('f')
        osutils.delete_any('f')
        osutils.delete_any('d')