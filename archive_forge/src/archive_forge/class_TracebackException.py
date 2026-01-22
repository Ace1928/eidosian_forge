import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
class TracebackException:

    def __init__(self, *args, **kwargs):
        self.capture_locals = kwargs.get('capture_locals', False)

    def format(self):
        result = ['A traceback']
        if self.capture_locals:
            result.append('locals')
        return result