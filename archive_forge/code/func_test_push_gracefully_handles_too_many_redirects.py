import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_gracefully_handles_too_many_redirects(self):
    """Push fails gracefully if the mkdir generates a large number of
        redirects.
        """
    destination_url = self.memory_server.get_url() + 'infinite-loop'
    out, err = self.run_bzr_error(['Too many redirections trying to make %s\\.\n' % re.escape(destination_url)], ['push', '-d', 'tree', destination_url], retcode=3)
    self.assertEqual('', out)