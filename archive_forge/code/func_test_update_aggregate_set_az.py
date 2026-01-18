from openstack.tests import fakes
from openstack.tests.unit import base
def test_update_aggregate_set_az(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['os-aggregates', '1']), json=self.fake_aggregate), dict(method='PUT', uri=self.get_mock_url('compute', 'public', append=['os-aggregates', '1']), json={'aggregate': self.fake_aggregate}, validate=dict(json={'aggregate': {'availability_zone': 'az'}}))])
    self.cloud.update_aggregate(1, availability_zone='az')
    self.assert_calls()