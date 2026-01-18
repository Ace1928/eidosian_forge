import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_purge_node_instance_info(self):
    test_patch = [{'op': 'remove', 'path': '/instance_info'}]
    self.fake_baremetal_node['instance_info'] = {}
    self.register_uris([dict(method='PATCH', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node, validate=dict(json=test_patch))])
    self.cloud.purge_node_instance_info(self.fake_baremetal_node['uuid'])
    self.assert_calls()