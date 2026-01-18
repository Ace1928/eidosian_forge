from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_gateway_ip_qos(self):
    qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()
    self.network_client.find_qos_policy = mock.Mock(return_value=qos_policy)
    arglist = ['--external-gateway', self._network.id, '--qos-policy', qos_policy.id, self._router.id]
    verifylist = [('router', self._router.id), ('external_gateway', self._network.id), ('qos_policy', qos_policy.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.update_router.assert_called_with(self._router, **{'external_gateway_info': {'network_id': self._network.id, 'qos_policy_id': qos_policy.id}})
    self.assertIsNone(result)