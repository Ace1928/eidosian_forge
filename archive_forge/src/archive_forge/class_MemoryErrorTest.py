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
class MemoryErrorTest(unittest.TestCase):

    def test_memoryerror(self):
        memoryerror_code = '(' * 200 + ')' * 200
        ip.run_cell(memoryerror_code)