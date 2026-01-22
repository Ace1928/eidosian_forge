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
def _try_loading(self):
    try:
        import breezy._fictional_extension_py
    except ImportError as e:
        osutils.failed_to_load_extension(e)
        return True