import threading
from unittest import mock
import warnings
import eventlet
from eventlet import greenthread
from oslotest import base as test_base
from oslo_utils import eventletutils
def thread_func():
    result = event.wait(0.2)
    wakes.append(result)
    if len(wakes) == 1:
        self.assertTrue(result)
        event.clear()
    else:
        self.assertFalse(result)