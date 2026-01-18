import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_unregister_machine_unavailable(self):
    invalid_states = ['active', 'cleaning', 'clean wait', 'clean failed']
    mac_address = self.fake_baremetal_port['address']
    nics = [{'mac': mac_address}]
    url_list = []
    for state in invalid_states:
        self.fake_baremetal_node['provision_state'] = state
        url_list.append(dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node))
    self.register_uris(url_list)
    for state in invalid_states:
        self.assertRaises(exceptions.SDKException, self.cloud.unregister_machine, nics, self.fake_baremetal_node['uuid'])
    self.assert_calls()