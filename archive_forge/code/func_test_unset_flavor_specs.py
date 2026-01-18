from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_unset_flavor_specs(self):
    self.use_compute_discovery()
    keys = ['key1', 'key2']
    self.register_uris([dict(method='DELETE', uri='{endpoint}/flavors/{id}/os-extra_specs/{key}'.format(endpoint=fakes.COMPUTE_ENDPOINT, id=1, key=key)) for key in keys])
    self.cloud.unset_flavor_specs(1, keys)
    self.assert_calls()