import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_omit_merges_with_sidelines(self):
    self.assertLogRevnos(['--omit-merges', '-n0'], ['1.2.1', '1.1.1', '1'], working_dir='level0')