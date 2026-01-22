import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
import testtools
from openstackclient.network.v2 import network_trunk
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
class TestDeleteNetworkTrunk(TestNetworkTrunk):
    project = identity_fakes_v3.FakeProject.create_one_project()
    domain = identity_fakes_v3.FakeDomain.create_one_domain()
    trunk_networks = network_fakes.create_networks(count=2)
    parent_port = network_fakes.create_one_port(attrs={'project_id': project.id, 'network_id': trunk_networks[0]['id']})
    sub_port = network_fakes.create_one_port(attrs={'project_id': project.id, 'network_id': trunk_networks[1]['id']})
    new_trunks = network_fakes.create_trunks(attrs={'project_id': project.id, 'port_id': parent_port['id'], 'sub_ports': {'port_id': sub_port['id'], 'segmentation_id': 42, 'segmentation_type': 'vlan'}})

    def setUp(self):
        super().setUp()
        self.network_client.find_trunk = mock.Mock(side_effect=[self.new_trunks[0], self.new_trunks[1]])
        self.network_client.delete_trunk = mock.Mock(return_value=None)
        self.network_client.find_port = mock.Mock(side_effect=[self.parent_port, self.sub_port])
        self.projects_mock.get.return_value = self.project
        self.domains_mock.get.return_value = self.domain
        self.cmd = network_trunk.DeleteNetworkTrunk(self.app, self.namespace)

    def test_delete_trunkx(self):
        arglist = [self.new_trunks[0].name]
        verifylist = [('trunk', [self.new_trunks[0].name])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.delete_trunk.assert_called_once_with(self.new_trunks[0].id)
        self.assertIsNone(result)

    def test_delete_trunk_multiple(self):
        arglist = []
        verifylist = []
        for t in self.new_trunks:
            arglist.append(t['name'])
        verifylist = [('trunk', arglist)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        calls = []
        for t in self.new_trunks:
            calls.append(call(t.id))
        self.network_client.delete_trunk.assert_has_calls(calls)
        self.assertIsNone(result)

    def test_delete_trunk_multiple_with_exception(self):
        arglist = [self.new_trunks[0].name, 'unexist_trunk']
        verifylist = [('trunk', [self.new_trunks[0].name, 'unexist_trunk'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.network_client.find_trunk = mock.Mock(side_effect=[self.new_trunks[0], exceptions.CommandError])
        with testtools.ExpectedException(exceptions.CommandError) as e:
            self.cmd.take_action(parsed_args)
            self.assertEqual('1 of 2 trunks failed to delete.', str(e))
        self.network_client.delete_trunk.assert_called_once_with(self.new_trunks[0].id)