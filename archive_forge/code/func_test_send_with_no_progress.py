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
def test_send_with_no_progress(self):

    class NoSendingSocket:

        def __init__(self):
            self.call_count = 0

        def send(self, bytes):
            self.call_count += 1
            if self.call_count > 100:
                raise RuntimeError('too many calls')
            return 0
    sock = NoSendingSocket()
    self.assertRaises(errors.ConnectionReset, osutils.send_all, sock, b'content')
    self.assertEqual(1, sock.call_count)