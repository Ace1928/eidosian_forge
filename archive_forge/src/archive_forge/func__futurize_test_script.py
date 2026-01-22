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
def _futurize_test_script(self, filename='mytestscript.py', stages=(1, 2), all_imports=False, from3=False, conservative=False):
    params = []
    stages = list(stages)
    if all_imports:
        params.append('--all-imports')
    if from3:
        script = 'pasteurize.py'
    else:
        script = 'futurize.py'
        if stages == [1]:
            params.append('--stage1')
        elif stages == [2]:
            params.append('--stage2')
        else:
            assert stages == [1, 2]
        if conservative:
            params.append('--conservative')
    fn = self.tempdir + filename
    call_args = [sys.executable, script] + params + ['-w', fn]
    try:
        output = check_output(call_args, stderr=STDOUT, env=self.env)
    except CalledProcessError as e:
        with open(fn) as f:
            msg = 'Error running the command %s\n%s\nContents of file %s:\n\n%s' % (' '.join(call_args), 'env=%s' % self.env, fn, '----\n%s\n----' % f.read())
        ErrorClass = FuturizeError if 'futurize' in script else PasteurizeError
        if not hasattr(e, 'output'):
            e.output = None
        raise ErrorClass(msg, e.returncode, e.cmd, output=e.output)
    return output