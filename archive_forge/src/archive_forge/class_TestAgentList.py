from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.compute.v2 import agent
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestAgentList(TestAgent):
    agents = compute_fakes.create_agents(count=3)
    list_columns = ('Agent ID', 'Hypervisor', 'OS', 'Architecture', 'Version', 'Md5Hash', 'URL')
    list_data = []
    for _agent in agents:
        list_data.append((_agent.agent_id, _agent.hypervisor, _agent.os, _agent.architecture, _agent.version, _agent.md5hash, _agent.url))

    def setUp(self):
        super(TestAgentList, self).setUp()
        self.agents_mock.list.return_value = self.agents
        self.cmd = agent.ListAgent(self.app, None)

    def test_agent_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.list_columns, columns)
        self.assertEqual(self.list_data, list(data))

    def test_agent_list_with_hypervisor(self):
        arglist = ['--hypervisor', 'hypervisor']
        verifylist = [('hypervisor', 'hypervisor')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.list_columns, columns)
        self.assertEqual(self.list_data, list(data))