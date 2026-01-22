import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
class BufferedWriter:

    def __init__(self):
        self.result = ''
        self.buffer = ''

    def write(self, arg):
        self.buffer += arg

    def flush(self):
        self.result += self.buffer
        self.buffer = ''

    def getvalue(self):
        return self.result