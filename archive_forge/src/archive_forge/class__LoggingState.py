from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
class _LoggingState(object):
    """
    State that helps to provide a reentrant gdb.execute() function.
    """

    def __init__(self):
        f = tempfile.NamedTemporaryFile('r+')
        self.file = f
        self.filename = f.name
        self.fd = f.fileno()
        _execute('set logging file %s' % self.filename)
        self.file_position_stack = []

    def __enter__(self):
        if not self.file_position_stack:
            _execute('set logging redirect on')
            _execute('set logging on')
            _execute('set pagination off')
        self.file_position_stack.append(os.fstat(self.fd).st_size)
        return self

    def getoutput(self):
        gdb.flush()
        self.file.seek(self.file_position_stack[-1])
        result = self.file.read()
        return result

    def __exit__(self, exc_type, exc_val, tb):
        startpos = self.file_position_stack.pop()
        self.file.seek(startpos)
        self.file.truncate()
        if not self.file_position_stack:
            _execute('set logging off')
            _execute('set logging redirect off')
            _execute('set pagination on')