import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_wait_for_baremetal_node_lock_timeout(self):
    self.fake_baremetal_node['reservation'] = 'conductor0'
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node)])
    self.assertRaises(exceptions.SDKException, self.cloud.wait_for_baremetal_node_lock, self.fake_baremetal_node, timeout=0.001)
    self.assert_calls()