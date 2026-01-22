from unittest import mock
from unittest.mock import call
import ddt
from osc_lib import exceptions
from openstackclient.network.v2 import network_rbac
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestSetNetworkRBAC(TestNetworkRBAC):
    project = identity_fakes_v3.FakeProject.create_one_project()
    rbac_policy = network_fakes.create_one_network_rbac(attrs={'target_tenant': project.id})

    def setUp(self):
        super(TestSetNetworkRBAC, self).setUp()
        self.cmd = network_rbac.SetNetworkRBAC(self.app, self.namespace)
        self.network_client.find_rbac_policy = mock.Mock(return_value=self.rbac_policy)
        self.network_client.update_rbac_policy = mock.Mock(return_value=None)
        self.projects_mock.get.return_value = self.project

    def test_network_rbac_set_nothing(self):
        arglist = [self.rbac_policy.id]
        verifylist = [('rbac_policy', self.rbac_policy.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.find_rbac_policy.assert_called_once_with(self.rbac_policy.id, ignore_missing=False)
        attrs = {}
        self.network_client.update_rbac_policy.assert_called_once_with(self.rbac_policy, **attrs)
        self.assertIsNone(result)

    def test_network_rbac_set(self):
        arglist = ['--target-project', self.project.id, self.rbac_policy.id]
        verifylist = [('target_project', self.project.id), ('rbac_policy', self.rbac_policy.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.find_rbac_policy.assert_called_once_with(self.rbac_policy.id, ignore_missing=False)
        attrs = {'target_tenant': self.project.id}
        self.network_client.update_rbac_policy.assert_called_once_with(self.rbac_policy, **attrs)
        self.assertIsNone(result)