import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
class TestLogVerbose(TestLog):

    def setUp(self):
        super().setUp()
        self.make_minimal_branch()

    def assertUseShortDeltaFormat(self, cmd):
        log = self.run_bzr(cmd)[0]
        self.assertContainsRe(log, '(?m)^\\s*A  hello.txt$')
        self.assertNotContainsRe(log, '(?m)^\\s*added:$')

    def assertUseLongDeltaFormat(self, cmd):
        log = self.run_bzr(cmd)[0]
        self.assertNotContainsRe(log, '(?m)^\\s*A  hello.txt$')
        self.assertContainsRe(log, '(?m)^\\s*added:$')

    def test_log_short_verbose(self):
        self.assertUseShortDeltaFormat(['log', '--short', '-v'])

    def test_log_s_verbose(self):
        self.assertUseShortDeltaFormat(['log', '-S', '-v'])

    def test_log_short_verbose_verbose(self):
        self.assertUseLongDeltaFormat(['log', '--short', '-vv'])

    def test_log_long_verbose(self):
        self.assertUseLongDeltaFormat(['log', '--long', '-v'])

    def test_log_long_verbose_verbose(self):
        self.assertUseLongDeltaFormat(['log', '--long', '-vv'])