import contextlib
import unittest
from unittest import mock
import eventlet
from eventlet import debug as eventlet_debug
from eventlet import greenpool
from oslo_log import pipe_mutex
def test_release_without_acquire(self):
    self.assertRaises(RuntimeError, self.mutex.release)