import sys
import unittest
from unittest import mock
from bpython.curtsiesfrontend.coderunner import CodeRunner, FakeOutput
def test_simple(self):
    c = CodeRunner(request_refresh=lambda: self.orig_stdout.flush() or self.orig_stderr.flush())
    stdout = FakeOutput(c, lambda *args, **kwargs: None, None)
    stderr = FakeOutput(c, lambda *args, **kwargs: None, None)
    sys.stdout = stdout
    sys.stdout = stderr
    c.load_code('1 + 1')
    c.run_code()
    c.run_code()
    c.run_code()