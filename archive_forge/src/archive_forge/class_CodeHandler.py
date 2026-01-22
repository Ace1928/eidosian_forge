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
class CodeHandler(unittest.TestCase):
    """
    Handy mixin for test classes for writing / reading / futurizing /
    running .py files in the test suite.
    """

    def setUp(self):
        """
        The outputs from the various futurize stages should have the
        following headers:
        """
        self.headers1 = reformat_code('\n        from __future__ import absolute_import\n        from __future__ import division\n        from __future__ import print_function\n        ')
        self.headers2 = reformat_code('\n        from __future__ import absolute_import\n        from __future__ import division\n        from __future__ import print_function\n        from __future__ import unicode_literals\n        from future import standard_library\n        standard_library.install_aliases()\n        from builtins import *\n        ')
        self.interpreters = [sys.executable]
        self.tempdir = tempfile.mkdtemp() + os.path.sep
        pypath = os.getenv('PYTHONPATH')
        if pypath:
            self.env = {'PYTHONPATH': os.getcwd() + os.pathsep + pypath}
        else:
            self.env = {'PYTHONPATH': os.getcwd()}

    def convert(self, code, stages=(1, 2), all_imports=False, from3=False, reformat=True, run=True, conservative=False):
        """
        Converts the code block using ``futurize`` and returns the
        resulting code.

        Passing stages=[1] or stages=[2] passes the flag ``--stage1`` or
        ``stage2`` to ``futurize``. Passing both stages runs ``futurize``
        with both stages by default.

        If from3 is False, runs ``futurize``, converting from Python 2 to
        both 2 and 3. If from3 is True, runs ``pasteurize`` to convert
        from Python 3 to both 2 and 3.

        Optionally reformats the code block first using the reformat() function.

        If run is True, runs the resulting code under all Python
        interpreters in self.interpreters.
        """
        if reformat:
            code = reformat_code(code)
        self._write_test_script(code)
        self._futurize_test_script(stages=stages, all_imports=all_imports, from3=from3, conservative=conservative)
        output = self._read_test_script()
        if run:
            for interpreter in self.interpreters:
                _ = self._run_test_script(interpreter=interpreter)
        return output

    def compare(self, output, expected, ignore_imports=True):
        """
        Compares whether the code blocks are equal. If not, raises an
        exception so the test fails. Ignores any trailing whitespace like
        blank lines.

        If ignore_imports is True, passes the code blocks into the
        strip_future_imports method.

        If one code block is a unicode string and the other a
        byte-string, it assumes the byte-string is encoded as utf-8.
        """
        if ignore_imports:
            output = self.strip_future_imports(output)
            expected = self.strip_future_imports(expected)
        if isinstance(output, bytes) and (not isinstance(expected, bytes)):
            output = output.decode('utf-8')
        if isinstance(expected, bytes) and (not isinstance(output, bytes)):
            expected = expected.decode('utf-8')
        self.assertEqual(order_future_lines(output.rstrip()), expected.rstrip())

    def strip_future_imports(self, code):
        """
        Strips any of these import lines:

            from __future__ import <anything>
            from future <anything>
            from future.<anything>
            from builtins <anything>

        or any line containing:
            install_hooks()
        or:
            install_aliases()

        Limitation: doesn't handle imports split across multiple lines like
        this:

            from __future__ import (absolute_import, division, print_function,
                                    unicode_literals)
        """
        output = []
        for line in code.split('\n'):
            if not (line.startswith('from __future__ import ') or line.startswith('from future ') or line.startswith('from builtins ') or ('install_hooks()' in line) or ('install_aliases()' in line) or line.startswith('from future.')):
                output.append(line)
        return '\n'.join(output)

    def convert_check(self, before, expected, stages=(1, 2), all_imports=False, ignore_imports=True, from3=False, run=True, conservative=False):
        """
        Convenience method that calls convert() and compare().

        Reformats the code blocks automatically using the reformat_code()
        function.

        If all_imports is passed, we add the appropriate import headers
        for the stage(s) selected to the ``expected`` code-block, so they
        needn't appear repeatedly in the test code.

        If ignore_imports is True, ignores the presence of any lines
        beginning:

            from __future__ import ...
            from future import ...

        for the purpose of the comparison.
        """
        output = self.convert(before, stages=stages, all_imports=all_imports, from3=from3, run=run, conservative=conservative)
        if all_imports:
            headers = self.headers2 if 2 in stages else self.headers1
        else:
            headers = ''
        reformatted = reformat_code(expected)
        if headers in reformatted:
            headers = ''
        self.compare(output, headers + reformatted, ignore_imports=ignore_imports)

    def unchanged(self, code, **kwargs):
        """
        Convenience method to ensure the code is unchanged by the
        futurize process.
        """
        self.convert_check(code, code, **kwargs)

    def _write_test_script(self, code, filename='mytestscript.py'):
        """
        Dedents the given code (a multiline string) and writes it out to
        a file in a temporary folder like /tmp/tmpUDCn7x/mytestscript.py.
        """
        if isinstance(code, bytes):
            code = code.decode('utf-8')
        with io.open(self.tempdir + filename, 'wt', encoding='utf-8') as f:
            f.write(dedent(code))

    def _read_test_script(self, filename='mytestscript.py'):
        with io.open(self.tempdir + filename, 'rt', encoding='utf-8') as f:
            newsource = f.read()
        return newsource

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