import os
import sys
from breezy import osutils, tests, urlutils
from breezy.tests import EncodingAdapter
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_remove_verbose(self):
    fname = self.info['filename']
    txt = self.run_bzr_decode(['remove', '--verbose', fname], encoding='ascii')