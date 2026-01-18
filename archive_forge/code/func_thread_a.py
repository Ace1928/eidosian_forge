import threading
from unittest import mock
import warnings
import eventlet
from eventlet import greenthread
from oslotest import base as test_base
from oslo_utils import eventletutils
def thread_a():
    self.assertFalse(event.wait(0.5))