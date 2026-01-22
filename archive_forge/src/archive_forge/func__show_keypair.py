import tempfile
from tempest.lib import exceptions
from novaclient.tests.functional import base
from novaclient.tests.functional.v2 import fake_crypto
def _show_keypair(self, key_name):
    return self.nova('keypair-show %s' % key_name)