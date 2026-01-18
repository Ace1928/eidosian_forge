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
@recursionlimit(160)
def test_recursion_three_frames(self):
    with tt.AssertPrints('[... skipping similar frames: '), tt.AssertPrints(re.compile('r3a at line 8 \\(\\d{2} times\\)'), suppress=False), tt.AssertPrints(re.compile('r3b at line 11 \\(\\d{2} times\\)'), suppress=False), tt.AssertPrints(re.compile('r3c at line 14 \\(\\d{2} times\\)'), suppress=False):
        ip.run_cell('r3o2()')