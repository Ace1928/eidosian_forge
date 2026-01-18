import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_register_machine_port_create_failed(self):
    mac_address = '00:01:02:03:04:05'
    nics = [{'address': mac_address}]
    node_uuid = self.fake_baremetal_node['uuid']
    node_to_post = {'chassis_uuid': None, 'driver': None, 'driver_info': None, 'name': self.fake_baremetal_node['name'], 'properties': None, 'uuid': node_uuid}
    self.fake_baremetal_node['provision_state'] = 'available'
    self.register_uris([dict(method='POST', uri=self.get_mock_url(resource='nodes'), json=self.fake_baremetal_node, validate=dict(json=node_to_post)), dict(method='POST', uri=self.get_mock_url(resource='ports'), status_code=400, json={'error': 'no ports for you'}, validate=dict(json={'address': mac_address, 'node_uuid': node_uuid})), dict(method='DELETE', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]))])
    self.assertRaisesRegex(exceptions.SDKException, 'no ports for you', self.cloud.register_machine, nics, **node_to_post)
    self.assert_calls()