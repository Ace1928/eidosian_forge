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
def test_plain_reports_notes(self):
    with tt.AssertPrints(['AssertionError', 'Message', 'This is a PEP-678 note.']):
        ip.run_cell('%xmode Plain')
        ip.run_cell(self.ERROR_WITH_NOTE)
        ip.run_cell('%xmode Verbose')