from unittest import mock
from unittest.mock import call
import ddt
from osc_lib import exceptions
from openstackclient.network.v2 import network_rbac
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowNetworkRBAC(TestNetworkRBAC):
    rbac_policy = network_fakes.create_one_network_rbac()
    columns = ('action', 'id', 'object_id', 'object_type', 'project_id', 'target_project_id')
    data = [rbac_policy.action, rbac_policy.id, rbac_policy.object_id, rbac_policy.object_type, rbac_policy.project_id, rbac_policy.target_project_id]

    def setUp(self):
        super(TestShowNetworkRBAC, self).setUp()
        self.cmd = network_rbac.ShowNetworkRBAC(self.app, self.namespace)
        self.network_client.find_rbac_policy = mock.Mock(return_value=self.rbac_policy)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_network_rbac_show_all_options(self):
        arglist = [self.rbac_policy.id]
        verifylist = [('rbac_policy', self.rbac_policy.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_rbac_policy.assert_called_with(self.rbac_policy.id, ignore_missing=False)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))