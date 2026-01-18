import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_wait_for_baremetal_node_lock_locked(self):
    self.fake_baremetal_node['reservation'] = 'conductor0'
    unlocked_node = self.fake_baremetal_node.copy()
    unlocked_node['reservation'] = None
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node), dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=unlocked_node)])
    self.assertIsNone(self.cloud.wait_for_baremetal_node_lock(self.fake_baremetal_node, timeout=1))
    self.assert_calls()