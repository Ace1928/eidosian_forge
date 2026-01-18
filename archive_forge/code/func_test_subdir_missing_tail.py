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
def test_subdir_missing_tail(self):
    self.build_tree(['MixedCaseParent/', 'MixedCaseParent/a_child'])
    base = osutils.realpath(self.get_transport('.').local_abspath('.'))
    self.assertRelpath('MixedCaseParent/a_child', base, 'MixedCaseParent/a_child')
    self.assertRelpath('MixedCaseParent/a_child', base, 'MixedCaseParent/A_Child')
    self.assertRelpath('MixedCaseParent/not_child', base, 'MixedCaseParent/not_child')