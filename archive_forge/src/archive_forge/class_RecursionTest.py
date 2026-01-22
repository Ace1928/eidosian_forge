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
class RecursionTest(unittest.TestCase):
    DEFINITIONS = '\ndef non_recurs():\n    1/0\n\ndef r1():\n    r1()\n\ndef r3a():\n    r3b()\n\ndef r3b():\n    r3c()\n\ndef r3c():\n    r3a()\n\ndef r3o1():\n    r3a()\n\ndef r3o2():\n    r3o1()\n'

    def setUp(self):
        ip.run_cell(self.DEFINITIONS)

    def test_no_recursion(self):
        with tt.AssertNotPrints('skipping similar frames'):
            ip.run_cell('non_recurs()')

    @recursionlimit(200)
    def test_recursion_one_frame(self):
        with tt.AssertPrints(re.compile('\\[\\.\\.\\. skipping similar frames: r1 at line 5 \\(\\d{2,3} times\\)\\]')):
            ip.run_cell('r1()')

    @recursionlimit(160)
    def test_recursion_three_frames(self):
        with tt.AssertPrints('[... skipping similar frames: '), tt.AssertPrints(re.compile('r3a at line 8 \\(\\d{2} times\\)'), suppress=False), tt.AssertPrints(re.compile('r3b at line 11 \\(\\d{2} times\\)'), suppress=False), tt.AssertPrints(re.compile('r3c at line 14 \\(\\d{2} times\\)'), suppress=False):
            ip.run_cell('r3o2()')