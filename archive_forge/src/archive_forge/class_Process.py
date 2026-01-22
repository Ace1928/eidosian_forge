import multiprocessing
import os
import platform
import sys
import unittest
from absl import app
from absl import logging
from tensorflow.python.eager import test
class Process(object):
    """A process that skips test (until windows is supported)."""

    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise unittest.SkipTest('TODO(b/150264776): Windows is not supported in MultiProcessRunner.')