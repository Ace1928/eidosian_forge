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
def test_syntaxerror_stacktrace_when_running_compiled_code(self):
    syntax_error_at_runtime = '\ndef foo():\n    eval("..")\n\ndef bar():\n    foo()\n\nbar()\n'
    with tt.AssertPrints('SyntaxError'):
        ip.run_cell(syntax_error_at_runtime)
    with tt.AssertPrints(['foo()', 'bar()']):
        ip.run_cell(syntax_error_at_runtime)
    del ip.user_ns['bar']
    del ip.user_ns['foo']