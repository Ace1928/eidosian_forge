import os
from breezy import branch, builtins, errors
from breezy.tests import transport_util
class TestCommitWithBoundBranch(transport_util.TestCaseWithConnectionHookedTransport):

    def setUp(self):
        super().setUp()
        self.master_wt = self.make_branch_and_tree('master')
        self.local_wt = self.make_branch_and_tree('local')
        master_branch = branch.Branch.open(self.get_url('master'))
        self.local_wt.branch.bind(master_branch)

    def test_commit_mine_modified(self):
        self.start_logging_connections()
        commit = builtins.cmd_commit()
        os.chdir('local')
        commit.run(message='empty commit', unchanged=True)
        self.assertEqual(1, len(self.connections))

    def test_commit_both_modified(self):
        self.master_wt.commit('empty commit on master')
        self.start_logging_connections()
        commit = builtins.cmd_commit()
        os.chdir('local')
        self.assertRaises(errors.BoundBranchOutOfDate, commit.run, message='empty commit', unchanged=True)
        self.assertEqual(1, len(self.connections))

    def test_commit_local(self):
        """Commits with --local should not connect to the master!"""
        self.start_logging_connections()
        commit = builtins.cmd_commit()
        os.chdir('local')
        commit.run(message='empty commit', unchanged=True, local=True)
        self.assertEqual(0, len(self.connections))