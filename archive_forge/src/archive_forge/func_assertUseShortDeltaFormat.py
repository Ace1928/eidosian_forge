import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def assertUseShortDeltaFormat(self, cmd):
    log = self.run_bzr(cmd)[0]
    self.assertContainsRe(log, '(?m)^\\s*A  hello.txt$')
    self.assertNotContainsRe(log, '(?m)^\\s*added:$')