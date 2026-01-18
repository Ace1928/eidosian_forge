import atexit
import functools
import numpy
import os
import random
import types
import unittest
import cupy
def wrap_tearDown(f):

    def func(self):
        try:
            f(self)
        finally:
            _teardown_random()
    return func