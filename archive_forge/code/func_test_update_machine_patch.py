import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_update_machine_patch(self):
    if self.field_name not in self.fake_baremetal_node:
        self.fake_baremetal_node[self.field_name] = None
    value_to_send = self.fake_baremetal_node[self.field_name]
    if self.changed:
        value_to_send = self.new_value
    uris = [dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node)]
    if self.changed:
        test_patch = [{'op': 'replace', 'path': '/' + self.field_name, 'value': value_to_send}]
        uris.append(dict(method='PATCH', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node, validate=dict(json=test_patch)))
    self.register_uris(uris)
    call_args = {self.field_name: value_to_send}
    update_dict = self.cloud.update_machine(self.fake_baremetal_node['uuid'], **call_args)
    if self.changed:
        self.assertEqual(['/' + self.field_name], update_dict['changes'])
    else:
        self.assertIsNone(update_dict['changes'])
    self.assertSubdict(self.fake_baremetal_node, update_dict['node'])
    self.assert_calls()