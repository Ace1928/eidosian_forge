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
def test_option_concurrency(self):
    self.overrideEnv('BRZ_CONCURRENCY', '1')
    self.run_bzr('rocks --concurrency 42')
    self.assertEqual('42', os.environ['BRZ_CONCURRENCY'])
    self.assertEqual(42, osutils.local_concurrency(use_cache=False))