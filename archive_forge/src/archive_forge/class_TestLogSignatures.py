import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
class TestLogSignatures(TestLog):

    def test_log_with_signatures(self):
        self.requireFeature(features.gpg)
        tree = self.make_linear_branch(format='dirstate-tags')
        log = self.run_bzr('log --signatures')[0]
        self.assertTrue('signature: no signature' in log)

    def test_log_without_signatures(self):
        self.requireFeature(features.gpg)
        tree = self.make_linear_branch(format='dirstate-tags')
        log = self.run_bzr('log')[0]
        self.assertFalse('signature: no signature' in log)