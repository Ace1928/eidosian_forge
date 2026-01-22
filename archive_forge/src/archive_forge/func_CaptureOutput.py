import contextlib
import os
import shutil
import sys
import tempfile
import unittest
import six
@contextlib.contextmanager
def CaptureOutput():
    new_stdout, new_stderr = (six.StringIO(), six.StringIO())
    old_stdout, old_stderr = (sys.stdout, sys.stderr)
    try:
        sys.stdout, sys.stderr = (new_stdout, new_stderr)
        yield (new_stdout, new_stderr)
    finally:
        sys.stdout, sys.stderr = (old_stdout, old_stderr)