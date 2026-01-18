from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_gateway_ip_qos_no_gateway(self):
    qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()
    self.network_client.find_qos_policy = mock.Mock(return_value=qos_policy)
    router = network_fakes.FakeRouter.create_one_router()
    self.network_client.find_router = mock.Mock(return_value=router)
    arglist = ['--qos-policy', qos_policy.id, router.id]
    verifylist = [('router', router.id), ('qos_policy', qos_policy.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)