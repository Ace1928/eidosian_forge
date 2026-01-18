import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
def test_update_with_description_longer_than_255_symbols(self):
    server = self._create_server()
    descr = ''.join((random.choice(string.ascii_letters) for i in range(256)))
    output = self.nova("update %s --description '%s'" % (server.id, descr), fail_ok=True, merge_stderr=True)
    self.assertIn("ERROR (BadRequest): Invalid input for field/attribute description. Value: %s. '%s' is too long (HTTP 400)" % (descr, descr), output)