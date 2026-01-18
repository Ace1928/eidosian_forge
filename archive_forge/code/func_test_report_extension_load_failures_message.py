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
def test_report_extension_load_failures_message(self):
    log = BytesIO()
    trace.push_log_file(log)
    self.assertTrue(self._try_loading())
    osutils.report_extension_load_failures()
    self.assertContainsRe(log.getvalue(), b'brz: warning: some compiled extensions could not be loaded; see ``brz help missing-extensions``\n')