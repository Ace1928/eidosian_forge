import codecs
import locale
import sys
from typing import Set
from .. import osutils
from . import TestCase
from .ui_testing import BytesIOWithEncoding, StringIOWithEncoding
def test_user_cp_unknown(self):
    self._encoding = 'cp-unknown'
    self.assertEqual('ascii', osutils.get_user_encoding())
    self.assertEqual('brz: warning: unknown encoding cp-unknown. Continuing with ascii encoding.\n', sys.stderr.getvalue())