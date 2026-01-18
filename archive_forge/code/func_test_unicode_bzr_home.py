import os
import sys
import breezy
from breezy import osutils, trace
from breezy.tests import (TestCase, TestCaseInTempDir, TestSkipped,
def test_unicode_bzr_home(self):
    uni_val, str_val = probe_unicode_in_user_encoding()
    if uni_val is None:
        raise TestSkipped('Cannot find a unicode character that works in encoding %s' % (osutils.get_user_encoding(),))
    self.overrideEnv('BRZ_HOME', uni_val)
    self.permit_source_tree_branch_repo()
    out = self.run_bzr_raw('version')[0]
    self.assertTrue(len(out) > 0)
    self.assertContainsRe(out, b'(?m)^  Breezy configuration: ' + str_val)