from gitdb import OStream
import sys
import random
from array import array
from io import BytesIO
import glob
import unittest
import tempfile
import shutil
import os
import gc
import logging
from functools import wraps
def with_rw_directory(func):
    """Create a temporary directory which can be written to, remove it if the
    test succeeds, but leave it otherwise to aid additional debugging"""

    def wrapper(self):
        path = tempfile.mktemp(prefix=func.__name__)
        os.mkdir(path)
        keep = False
        try:
            try:
                return func(self, path)
            except Exception:
                sys.stderr.write(f'Test {type(self).__name__}.{func.__name__} failed, output is at {path!r}\n')
                keep = True
                raise
        finally:
            if not keep:
                gc.collect()
                shutil.rmtree(path)
    wrapper.__name__ = func.__name__
    return wrapper