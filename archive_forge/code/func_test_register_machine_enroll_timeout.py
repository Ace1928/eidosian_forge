import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_register_machine_enroll_timeout(self):
    mac_address = '00:01:02:03:04:05'
    nics = [{'address': mac_address}]
    node_uuid = self.fake_baremetal_node['uuid']
    node_to_post = {'chassis_uuid': None, 'driver': None, 'driver_info': None, 'name': self.fake_baremetal_node['name'], 'properties': None, 'uuid': node_uuid}
    self.fake_baremetal_node['provision_state'] = 'enroll'
    busy_node = self.fake_baremetal_node.copy()
    busy_node['reservation'] = 'conductor0'
    busy_node['provision_state'] = 'verifying'
    self.register_uris([dict(method='POST', uri=self.get_mock_url(resource='nodes'), json=self.fake_baremetal_node, validate=dict(json=node_to_post)), dict(method='PUT', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid'], 'states', 'provision']), validate=dict(json={'target': 'manage'})), dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=busy_node), dict(method='DELETE', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]))])
    self.assertRaises(exceptions.SDKException, self.cloud.register_machine, nics, timeout=0.001, lock_timeout=0.001, **node_to_post)
    self.assert_calls()