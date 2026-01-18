import codecs
import locale
import sys
from typing import Set
from .. import osutils
from . import TestCase
from .ui_testing import BytesIOWithEncoding, StringIOWithEncoding
def test_user_cp0(self):
    self._encoding = 'cp0'
    self.assertEqual('ascii', osutils.get_user_encoding())
    self.assertEqual('', sys.stderr.getvalue())