import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
class TestLogWithLogCatcher(TestLog):

    def setUp(self):
        super().setUp()

        class MyLogFormatter(test_log.LogCatcher):

            def __new__(klass, *args, **kwargs):
                self.log_catcher = test_log.LogCatcher(*args, **kwargs)
                return self.log_catcher
        self.addCleanup(setattr, MyLogFormatter, '__new__', None)

        def getme(branch):
            return MyLogFormatter
        self.overrideAttr(log.log_formatter_registry, 'get_default', getme)

    def get_captured_revisions(self):
        return self.log_catcher.revisions

    def assertLogRevnos(self, args, expected_revnos, working_dir='.', out='', err=''):
        actual_out, actual_err = self.run_bzr(['log'] + args, working_dir=working_dir)
        self.assertEqual(out, actual_out)
        self.assertEqual(err, actual_err)
        self.assertEqual(expected_revnos, [r.revno for r in self.get_captured_revisions()])

    def assertLogRevnosAndDepths(self, args, expected_revnos_and_depths, working_dir='.'):
        self.run_bzr(['log'] + args, working_dir=working_dir)
        self.assertEqual(expected_revnos_and_depths, [(r.revno, r.merge_depth) for r in self.get_captured_revisions()])