import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def assertLogRevnosAndDepths(self, args, expected_revnos_and_depths, working_dir='.'):
    self.run_bzr(['log'] + args, working_dir=working_dir)
    self.assertEqual(expected_revnos_and_depths, [(r.revno, r.merge_depth) for r in self.get_captured_revisions()])