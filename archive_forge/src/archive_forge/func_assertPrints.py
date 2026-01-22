import contextlib
import imp
import inspect
import io
import sys
from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.impl import api
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
@contextlib.contextmanager
def assertPrints(self, expected_result):
    try:
        out_capturer = io.StringIO()
        sys.stdout = out_capturer
        yield
        self.assertEqual(out_capturer.getvalue(), expected_result)
    finally:
        sys.stdout = sys.__stdout__