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
def test_syntaxerror_no_stacktrace_at_compile_time(self):
    syntax_error_at_compile_time = '\ndef foo():\n    ..\n'
    with tt.AssertPrints('SyntaxError'):
        ip.run_cell(syntax_error_at_compile_time)
    with tt.AssertNotPrints('foo()'):
        ip.run_cell(syntax_error_at_compile_time)