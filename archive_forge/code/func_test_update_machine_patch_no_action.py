import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_update_machine_patch_no_action(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node)])
    update_dict = self.cloud.update_machine(self.fake_baremetal_node['uuid'])
    self.assertIsNone(update_dict['changes'])
    self.assertSubdict(self.fake_baremetal_node, update_dict['node'])
    self.assert_calls()