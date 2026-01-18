from openstack.tests import fakes
from openstack.tests.unit import base
def test_remove_host_from_aggregate(self):
    hostname = 'host1'
    self.register_uris([dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['os-aggregates', '1']), json=self.fake_aggregate), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['os-aggregates', '1', 'action']), json={'aggregate': self.fake_aggregate}, validate=dict(json={'remove_host': {'host': hostname}}))])
    self.cloud.remove_host_from_aggregate('1', hostname)
    self.assert_calls()