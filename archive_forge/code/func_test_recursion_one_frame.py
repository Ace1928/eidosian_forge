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
@recursionlimit(200)
def test_recursion_one_frame(self):
    with tt.AssertPrints(re.compile('\\[\\.\\.\\. skipping similar frames: r1 at line 5 \\(\\d{2,3} times\\)\\]')):
        ip.run_cell('r1()')