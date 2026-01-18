import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_register_machine_enroll_timeout_wait(self):
    mac_address = '00:01:02:03:04:05'
    nics = [{'address': mac_address}]
    node_uuid = self.fake_baremetal_node['uuid']
    node_to_post = {'chassis_uuid': None, 'driver': None, 'driver_info': None, 'name': self.fake_baremetal_node['name'], 'properties': None, 'uuid': node_uuid}
    self.fake_baremetal_node['provision_state'] = 'enroll'
    manageable_node = self.fake_baremetal_node.copy()
    manageable_node['provision_state'] = 'manageable'
    self.register_uris([dict(method='POST', uri=self.get_mock_url(resource='nodes'), json=self.fake_baremetal_node, validate=dict(json=node_to_post)), dict(method='PUT', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid'], 'states', 'provision']), validate=dict(json={'target': 'manage'})), dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=manageable_node), dict(method='POST', uri=self.get_mock_url(resource='ports'), validate=dict(json={'address': mac_address, 'node_uuid': node_uuid}), json=self.fake_baremetal_port), dict(method='PUT', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid'], 'states', 'provision']), validate=dict(json={'target': 'provide'})), dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node), dict(method='DELETE', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]))])
    self.assertRaises(exceptions.SDKException, self.cloud.register_machine, nics, wait=True, timeout=0.001, **node_to_post)
    self.assert_calls()