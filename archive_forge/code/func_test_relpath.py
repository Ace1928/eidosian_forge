import os
import sys
from breezy import osutils, tests, urlutils
from breezy.tests import EncodingAdapter
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_relpath(self):
    txt = self.run_bzr_decode(['relpath', self.info['filename']])
    self.assertEqual(self.info['filename'] + '\n', txt)
    self.run_bzr_decode(['relpath', self.info['filename']], encoding='ascii', fail=True)