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
def _run_test_script(self, filename='mytestscript.py', interpreter=sys.executable):
    fn = self.tempdir + filename
    try:
        output = check_output([interpreter, fn], env=self.env, stderr=STDOUT)
    except CalledProcessError as e:
        with open(fn) as f:
            msg = 'Error running the command %s\n%s\nContents of file %s:\n\n%s' % (' '.join([interpreter, fn]), 'env=%s' % self.env, fn, '----\n%s\n----' % f.read())
        if not hasattr(e, 'output'):
            e.output = None
        raise VerboseCalledProcessError(msg, e.returncode, e.cmd, output=e.output)
    return output