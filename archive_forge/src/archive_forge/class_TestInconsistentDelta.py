import os
from breezy import uncommit
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.errors import BoundBranchOutOfDate
from breezy.tests import TestCaseWithTransport
from breezy.tests.script import ScriptRunner, run_script
class TestInconsistentDelta(TestCaseWithTransport):

    def test_inconsistent_delta(self):
        wt = self.make_branch_and_tree('test')
        self.build_tree(['test/a/', 'test/a/b', 'test/a/c'])
        wt.add(['a', 'a/b', 'a/c'])
        wt.commit('initial commit', rev_id=b'a1')
        wt.remove(['a/b', 'a/c'])
        wt.commit('remove b and c', rev_id=b'a2')
        self.run_bzr('uncommit --force test')