import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
class TestPushRedirect(tests.TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        self.memory_server = RedirectingMemoryServer()
        self.start_server(self.memory_server)
        t = self.make_branch_and_tree('tree')
        self.build_tree(['tree/file'])
        t.add('file')
        t.commit('commit 1')

    def test_push_redirects_on_mkdir(self):
        """If the push requires a mkdir, push respects redirect requests.

        This is added primarily to handle lp:/ URI support, so that users can
        push to new branches by specifying lp:/ URIs.
        """
        destination_url = self.memory_server.get_url() + 'source'
        self.run_bzr(['push', '-d', 'tree', destination_url])
        local_revision = branch.Branch.open('tree').last_revision()
        remote_revision = branch.Branch.open(self.memory_server.get_url() + 'target').last_revision()
        self.assertEqual(remote_revision, local_revision)

    def test_push_gracefully_handles_too_many_redirects(self):
        """Push fails gracefully if the mkdir generates a large number of
        redirects.
        """
        destination_url = self.memory_server.get_url() + 'infinite-loop'
        out, err = self.run_bzr_error(['Too many redirections trying to make %s\\.\n' % re.escape(destination_url)], ['push', '-d', 'tree', destination_url], retcode=3)
        self.assertEqual('', out)