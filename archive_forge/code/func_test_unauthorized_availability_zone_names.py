from openstack.tests import fakes
from openstack.tests.unit import base
def test_unauthorized_availability_zone_names(self):
    self.register_uris([dict(method='GET', uri='{endpoint}/os-availability-zone'.format(endpoint=fakes.COMPUTE_ENDPOINT), status_code=403)])
    self.assertEqual([], self.cloud.list_availability_zone_names())
    self.assert_calls()