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
def test_report_extension_load_failures_no_warning(self):
    self.assertTrue(self._try_loading())
    warnings, result = self.callCatchWarnings(osutils.report_extension_load_failures)
    self.assertLength(0, warnings)