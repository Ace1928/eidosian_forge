import multiprocessing
import os
import platform
import sys
import unittest
from absl import app
from absl import logging
from tensorflow.python.eager import test
class AbslForkServerProcess(_AbslProcess, multiprocessing.context.ForkServerProcess):
    """An absl-compatible Forkserver process.

    Note: Forkserver is not available in windows.
    """