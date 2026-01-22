from unittest import mock
from openstackclient.network.v2 import network_auto_allocated_topology
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
class TestDeleteAutoAllocatedTopology(TestAutoAllocatedTopology):
    project = identity_fakes.FakeProject.create_one_project()
    network_object = network_fakes.create_one_network()
    topology = network_fakes.create_one_topology(attrs={'id': network_object.id, 'project_id': project.id})

    def setUp(self):
        super(TestDeleteAutoAllocatedTopology, self).setUp()
        self.cmd = network_auto_allocated_topology.DeleteAutoAllocatedTopology(self.app, self.namespace)
        self.network_client.delete_auto_allocated_topology = mock.Mock(return_value=None)

    def test_delete_no_project(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.delete_auto_allocated_topology.assert_called_once_with(None)
        self.assertIsNone(result)

    def test_delete_project_arg(self):
        arglist = ['--project', self.project.id]
        verifylist = [('project', self.project.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.delete_auto_allocated_topology.assert_called_once_with(self.project.id)
        self.assertIsNone(result)

    def test_delete_project_domain_arg(self):
        arglist = ['--project', self.project.id, '--project-domain', self.project.domain_id]
        verifylist = [('project', self.project.id), ('project_domain', self.project.domain_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.delete_auto_allocated_topology.assert_called_once_with(self.project.id)
        self.assertIsNone(result)