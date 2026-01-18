import atexit
import functools
import numpy
import os
import random
import types
import unittest
import cupy
def wrap_setUp(f):

    def func(self):
        _setup_random()
        f(self)
    return func