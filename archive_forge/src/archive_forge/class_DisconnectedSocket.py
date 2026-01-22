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
class DisconnectedSocket:

    def __init__(self, err):
        self.err = err

    def send(self, content):
        raise self.err

    def close(self):
        pass