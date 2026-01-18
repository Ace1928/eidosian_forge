import os
import sys
import breezy
from breezy import osutils, trace
from breezy.tests import (TestCase, TestCaseInTempDir, TestSkipped,
def test_unicode_brz_log(self):
    uni_val = '§'
    enc = osutils.get_user_encoding()
    try:
        str_val = uni_val.encode(enc)
    except UnicodeEncodeError:
        self.skipTest('Test string {!r} unrepresentable in user encoding {}'.format(uni_val, enc))
    brz_log = os.path.join(self.test_base_dir, uni_val)
    self.overrideEnv('BRZ_LOG', brz_log)
    out, err = self.run_brz_subprocess('version')
    uni_out = out.decode(enc)
    self.assertContainsRe(uni_out, '(?m)^  Breezy log file: .*/§$')