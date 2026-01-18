from openstack.tests import fakes
from openstack.tests.unit import base
def test_set_aggregate_metadata(self):
    metadata = {'key': 'value'}
    self.register_uris([dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['os-aggregates', '1']), json=self.fake_aggregate), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['os-aggregates', '1', 'action']), json={'aggregate': self.fake_aggregate}, validate=dict(json={'set_metadata': {'metadata': metadata}}))])
    self.cloud.set_aggregate_metadata('1', metadata)
    self.assert_calls()