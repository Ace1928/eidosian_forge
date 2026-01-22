import io
import os.path
import platform
import re
import sys
import traceback
import unittest
from textwrap import dedent
from tempfile import TemporaryDirectory
from IPython.core.ultratb import ColorTB, VerboseTB
from IPython.testing import tools as tt
from IPython.testing.decorators import onlyif_unicode_paths, skip_without
from IPython.utils.syspathcontext import prepended_to_syspath
import sys
class SyntaxErrorTest(unittest.TestCase):

    def test_syntaxerror_no_stacktrace_at_compile_time(self):
        syntax_error_at_compile_time = '\ndef foo():\n    ..\n'
        with tt.AssertPrints('SyntaxError'):
            ip.run_cell(syntax_error_at_compile_time)
        with tt.AssertNotPrints('foo()'):
            ip.run_cell(syntax_error_at_compile_time)

    def test_syntaxerror_stacktrace_when_running_compiled_code(self):
        syntax_error_at_runtime = '\ndef foo():\n    eval("..")\n\ndef bar():\n    foo()\n\nbar()\n'
        with tt.AssertPrints('SyntaxError'):
            ip.run_cell(syntax_error_at_runtime)
        with tt.AssertPrints(['foo()', 'bar()']):
            ip.run_cell(syntax_error_at_runtime)
        del ip.user_ns['bar']
        del ip.user_ns['foo']

    def test_changing_py_file(self):
        with TemporaryDirectory() as td:
            fname = os.path.join(td, 'foo.py')
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(se_file_1)
            with tt.AssertPrints(['7/', 'SyntaxError']):
                ip.magic('run ' + fname)
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(se_file_2)
            with tt.AssertPrints(['7/', 'SyntaxError']):
                ip.magic('run ' + fname)

    def test_non_syntaxerror(self):
        try:
            raise ValueError('QWERTY')
        except ValueError:
            with tt.AssertPrints('QWERTY'):
                ip.showsyntaxerror()