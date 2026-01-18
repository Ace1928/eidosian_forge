import os
import shutil
import tempfile
import unittest
import patiencediff
from . import _patiencediff_py
def test_patience_unified_diff_with_dates(self):
    txt_a = ['hello there\n', 'world\n', 'how are you today?\n']
    txt_b = ['hello there\n', 'how are you today?\n']
    unified_diff = patiencediff.unified_diff
    psm = self._PatienceSequenceMatcher
    self.assertEqual(['--- a\t2008-08-08\n', '+++ b\t2008-09-09\n', '@@ -1,3 +1,2 @@\n', ' hello there\n', '-world\n', ' how are you today?\n'], list(unified_diff(txt_a, txt_b, fromfile='a', tofile='b', fromfiledate='2008-08-08', tofiledate='2008-09-09', sequencematcher=psm)))