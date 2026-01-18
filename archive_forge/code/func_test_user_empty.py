import codecs
import locale
import sys
from typing import Set
from .. import osutils
from . import TestCase
from .ui_testing import BytesIOWithEncoding, StringIOWithEncoding
def test_user_empty(self):
    """Running bzr from a vim script gives '' for a preferred locale"""
    self._encoding = ''
    self.assertEqual('ascii', osutils.get_user_encoding())
    self.assertEqual('', sys.stderr.getvalue())