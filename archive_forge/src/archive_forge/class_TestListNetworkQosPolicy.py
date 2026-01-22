from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_policy
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestListNetworkQosPolicy(TestQosPolicy):
    qos_policies = network_fakes.FakeNetworkQosPolicy.create_qos_policies(count=3)
    columns = ('ID', 'Name', 'Shared', 'Default', 'Project')
    data = []
    for qos_policy in qos_policies:
        data.append((qos_policy.id, qos_policy.name, qos_policy.shared, qos_policy.is_default, qos_policy.project_id))

    def setUp(self):
        super(TestListNetworkQosPolicy, self).setUp()
        self.network_client.qos_policies = mock.Mock(return_value=self.qos_policies)
        self.cmd = network_qos_policy.ListNetworkQosPolicy(self.app, self.namespace)

    def test_qos_policy_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.qos_policies.assert_called_once_with(**{})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_qos_policy_list_share(self):
        arglist = ['--share']
        verifylist = [('share', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.qos_policies.assert_called_once_with(**{'shared': True})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_qos_policy_list_no_share(self):
        arglist = ['--no-share']
        verifylist = [('no_share', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.qos_policies.assert_called_once_with(**{'shared': False})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))

    def test_network_qos_list_project(self):
        project = identity_fakes_v3.FakeProject.create_one_project()
        self.projects_mock.get.return_value = project
        arglist = ['--project', project.id, '--project-domain', project.domain_id]
        verifylist = [('project', project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.qos_policies.assert_called_once_with(**{'project_id': project.id})
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))