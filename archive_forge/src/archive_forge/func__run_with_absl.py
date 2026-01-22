import multiprocessing
import os
import platform
import sys
import unittest
from absl import app
from absl import logging
from tensorflow.python.eager import test
def _run_with_absl(self):
    app.run(lambda _: self._run_impl())