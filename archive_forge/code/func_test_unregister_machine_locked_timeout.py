import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_unregister_machine_locked_timeout(self):
    mac_address = self.fake_baremetal_port['address']
    nics = [{'mac': mac_address}]
    self.fake_baremetal_node['provision_state'] = 'available'
    self.fake_baremetal_node['reservation'] = 'conductor99'
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node), dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node)])
    self.assertRaises(exceptions.SDKException, self.cloud.unregister_machine, nics, self.fake_baremetal_node['uuid'], timeout=0.001)
    self.assert_calls()