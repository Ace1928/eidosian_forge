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
class ChangedPyFileTest(unittest.TestCase):

    def test_changing_py_file(self):
        """Traceback produced if the line where the error occurred is missing?

        https://github.com/ipython/ipython/issues/1456
        """
        with TemporaryDirectory() as td:
            fname = os.path.join(td, 'foo.py')
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(file_1)
            with prepended_to_syspath(td):
                ip.run_cell('import foo')
            with tt.AssertPrints('ZeroDivisionError'):
                ip.run_cell('foo.f()')
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(file_2)
            with tt.AssertNotPrints('Internal Python error', channel='stderr'):
                with tt.AssertPrints('ZeroDivisionError'):
                    ip.run_cell('foo.f()')
                with tt.AssertPrints('ZeroDivisionError'):
                    ip.run_cell('foo.f()')