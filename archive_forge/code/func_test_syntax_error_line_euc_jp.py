import codecs
import datetime
import doctest
import io
from itertools import chain
from itertools import combinations
import os
import platform
from queue import Queue
import re
import shutil
import sys
import tempfile
import threading
from unittest import TestSuite
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.content_type import ContentType, UTF8_TEXT
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.tests.helpers import (
from testtools.testresult.doubles import (
from testtools.testresult.real import (
def test_syntax_error_line_euc_jp(self):
    """Syntax error on a euc_jp line shows the line decoded"""
    text, raw = self._get_sample_text('euc_jp')
    textoutput = self._setup_external_case('import bad')
    self._write_module('bad', 'euc_jp', '# coding: euc_jp\n$ = 0 # %s\n' % text)
    textoutput = self._run_external_case()
    if self._is_pypy:
        self._error_on_character = True
    self.assertIn(self._as_output(('    $ = 0 # %s\n' + ' ' * self._error_on_character + '   ^\nSyntaxError: ') % (text,)), textoutput)