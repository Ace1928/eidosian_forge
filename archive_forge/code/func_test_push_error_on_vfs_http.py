import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_error_on_vfs_http(self):
    """ pushing a branch to a HTTP server fails cleanly. """
    self.transport_readonly_server = http_server.HttpServer
    self.make_branch('source')
    public_url = self.get_readonly_url('target')
    self.run_bzr_error(['http does not support mkdir'], ['push', public_url], working_dir='source')