import gc
import io
import os
import sys
import signal
import weakref
import unittest
from test import support
def withRepeats(self, test_function, repeats=None):
    if not support.check_impl_detail(cpython=True):
        repeats = 1
    elif repeats is None:
        repeats = self.default_repeats
    for repeat in range(repeats):
        with self.subTest(repeat=repeat):
            if repeat != 0:
                self.setUp()
            try:
                test_function()
            finally:
                if repeat != repeats - 1:
                    self.tearDown()