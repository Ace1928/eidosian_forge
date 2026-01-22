from __future__ import print_function, absolute_import
import os
import tempfile
import unittest
import sys
import re
import warnings
import io
from textwrap import dedent
from future.utils import bind_method, PY26, PY3, PY2, PY27
from future.moves.subprocess import check_output, STDOUT, CalledProcessError
def _write_test_script(self, code, filename='mytestscript.py'):
    """
        Dedents the given code (a multiline string) and writes it out to
        a file in a temporary folder like /tmp/tmpUDCn7x/mytestscript.py.
        """
    if isinstance(code, bytes):
        code = code.decode('utf-8')
    with io.open(self.tempdir + filename, 'wt', encoding='utf-8') as f:
        f.write(dedent(code))