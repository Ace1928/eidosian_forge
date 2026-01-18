import unittest
import string
import numpy as np
from numba import njit, jit, literal_unroll
from numba.core import event as ev
from numba.tests.support import TestCase, override_config
def test_recording_listener(self):

    @njit
    def foo(x):
        return x + x
    with ev.install_recorder('numba:compile') as rec:
        foo(1)
    self.assertIsInstance(rec, ev.RecordingListener)
    self.assertGreaterEqual(len(rec.buffer), 2)