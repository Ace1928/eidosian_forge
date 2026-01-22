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
class _UTF8DirReaderFeature(features.ModuleAvailableFeature):

    def _probe(self):
        try:
            from .. import _readdir_pyx
            self._module = _readdir_pyx
            self.reader = _readdir_pyx.UTF8DirReader
            return True
        except ImportError:
            return False