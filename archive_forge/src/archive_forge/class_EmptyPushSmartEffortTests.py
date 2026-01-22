import os
from io import BytesIO
from ... import (branch, builtins, check, controldir, errors, push, revision,
from ...bzr import branch as bzrbranch
from ...bzr.smart import client
from .. import per_branch, test_server
class EmptyPushSmartEffortTests(per_branch.TestCaseWithBranch):
    """Tests that a push of 0 revisions should make a limited number of smart
    protocol RPCs.
    """

    def setUp(self):
        if self.transport_server is not None and issubclass(self.transport_server, test_server.SmartTCPServer_for_testing):
            raise tests.TestNotApplicable('Does not apply when remote backing branch is also a smart branch')
        if not self.branch_format.supports_leaving_lock():
            raise tests.TestNotApplicable('Branch format is not usable via HPSS.')
        super().setUp()
        self.smart_server = test_server.SmartTCPServer_for_testing()
        self.start_server(self.smart_server, self.get_server())
        self.empty_branch = self.make_branch('empty')
        self.make_branch('target')
        client._SmartClient.hooks.install_named_hook('call', self.capture_hpss_call, None)
        self.hpss_calls = []

    def capture_hpss_call(self, params):
        self.hpss_calls.append(params.method)

    def test_empty_branch_api(self):
        """The branch_obj.push API should make a limited number of HPSS calls.
        """
        t = transport.get_transport_from_url(self.smart_server.get_url()).clone('target')
        target = branch.Branch.open_from_transport(t)
        self.empty_branch.push(target)
        self.assertEqual([b'BzrDir.open_2.1', b'BzrDir.open_branchV3', b'BzrDir.find_repositoryV3', b'Branch.get_stacked_on_url', b'Branch.lock_write', b'Branch.last_revision_info', b'Branch.unlock'], self.hpss_calls)

    def test_empty_branch_command(self):
        """The 'bzr push' command should make a limited number of HPSS calls.
        """
        cmd = builtins.cmd_push()
        cmd.outf = BytesIO()
        cmd.run(directory=self.get_url('empty'), location=self.smart_server.get_url() + 'target')
        self.assertTrue(len(self.hpss_calls) <= 9, self.hpss_calls)