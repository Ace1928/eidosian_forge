import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_register_machine_enroll(self):
    mac_address = '00:01:02:03:04:05'
    nics = [{'address': mac_address, 'pxe_enabled': False}]
    node_uuid = self.fake_baremetal_node['uuid']
    node_to_post = {'chassis_uuid': None, 'driver': None, 'driver_info': None, 'name': self.fake_baremetal_node['name'], 'properties': None, 'uuid': node_uuid}
    self.fake_baremetal_node['provision_state'] = 'enroll'
    manageable_node = self.fake_baremetal_node.copy()
    manageable_node['provision_state'] = 'manageable'
    available_node = self.fake_baremetal_node.copy()
    available_node['provision_state'] = 'available'
    self.register_uris([dict(method='POST', uri=self.get_mock_url(resource='nodes'), validate=dict(json=node_to_post), json=self.fake_baremetal_node), dict(method='PUT', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid'], 'states', 'provision']), validate=dict(json={'target': 'manage'})), dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=manageable_node), dict(method='POST', uri=self.get_mock_url(resource='ports'), validate=dict(json={'address': mac_address, 'node_uuid': node_uuid, 'pxe_enabled': False}), json=self.fake_baremetal_port), dict(method='PUT', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid'], 'states', 'provision']), validate=dict(json={'target': 'provide'})), dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=available_node)])
    return_value = self.cloud.register_machine(nics, **node_to_post)
    self.assertSubdict(available_node, return_value)
    self.assert_calls()