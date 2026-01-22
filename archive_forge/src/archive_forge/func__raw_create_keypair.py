import tempfile
from tempest.lib import exceptions
from novaclient.tests.functional import base
from novaclient.tests.functional.v2 import fake_crypto
def _raw_create_keypair(self, **kwargs):
    key_name = self.name_generate()
    kwargs_str = self._serialize_kwargs(kwargs)
    self.nova('keypair-add %s %s' % (kwargs_str, key_name))
    return key_name