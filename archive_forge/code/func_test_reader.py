import asyncore
import errno
import socket
import threading
from taskflow.engines.action_engine import process_executor as pu
from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
def test_reader(self):
    capture_buf = []

    def do_capture(identity, message_capture_func):
        capture_buf.append(message_capture_func())
    r = pu.Reader(b'secret', do_capture)
    for data in pu._encode_message(b'secret', ['hi'], b'me'):
        self.assertEqual(len(data), r.bytes_needed)
        r.feed(data)
    self.assertEqual(1, len(capture_buf))
    self.assertEqual(['hi'], capture_buf[0])