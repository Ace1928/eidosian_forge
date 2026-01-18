import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_register_machine(self):
    mac_address = '00:01:02:03:04:05'
    nics = [{'address': mac_address}]
    node_uuid = self.fake_baremetal_node['uuid']
    node_to_post = {'driver': None, 'driver_info': None, 'name': self.fake_baremetal_node['name'], 'properties': None, 'uuid': node_uuid}
    self.fake_baremetal_node['provision_state'] = 'available'
    if 'provision_state' in node_to_post:
        node_to_post.pop('provision_state')
    self.register_uris([dict(method='POST', uri=self.get_mock_url(resource='nodes'), json=self.fake_baremetal_node, validate=dict(json=node_to_post)), dict(method='POST', uri=self.get_mock_url(resource='ports'), validate=dict(json={'address': mac_address, 'node_uuid': node_uuid}), json=self.fake_baremetal_port)])
    return_value = self.cloud.register_machine(nics, **node_to_post)
    self.assertEqual(self.uuid, return_value.id)
    self.assertSubdict(self.fake_baremetal_node, return_value)
    self.assert_calls()