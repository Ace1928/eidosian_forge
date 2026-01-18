import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_validate_machine_not_for_deploy(self):
    validate_return = {'deploy': {'result': False, 'reason': 'Not ready'}, 'power': {'result': True}, 'foo': {'result': False}}
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid'], 'validate']), json=validate_return)])
    self.cloud.validate_machine(self.fake_baremetal_node['uuid'], for_deploy=False)
    self.assert_calls()