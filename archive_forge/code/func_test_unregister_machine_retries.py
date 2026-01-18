import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_unregister_machine_retries(self):
    mac_address = self.fake_baremetal_port['address']
    nics = [{'mac': mac_address}]
    port_uuid = self.fake_baremetal_port['uuid']
    port_node_uuid = self.fake_baremetal_port['node_uuid']
    self.fake_baremetal_node['provision_state'] = 'available'
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node), dict(method='GET', uri=self.get_mock_url(resource='ports', qs_elements=['address=%s' % mac_address]), json={'ports': [{'address': mac_address, 'node_uuid': port_node_uuid, 'uuid': port_uuid}]}), dict(method='DELETE', status_code=503, uri=self.get_mock_url(resource='ports', append=[self.fake_baremetal_port['uuid']])), dict(method='DELETE', status_code=409, uri=self.get_mock_url(resource='ports', append=[self.fake_baremetal_port['uuid']])), dict(method='DELETE', uri=self.get_mock_url(resource='ports', append=[self.fake_baremetal_port['uuid']])), dict(method='DELETE', status_code=409, uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']])), dict(method='DELETE', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]))])
    self.cloud.unregister_machine(nics, self.fake_baremetal_node['uuid'])
    self.assert_calls()