from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils as tests_utils
import testtools
from neutronclient.osc.v2.sfc import sfc_service_graph
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestCreateSfcServiceGraph(fakes.TestNeutronClientOSCV2):
    _service_graph = fakes.FakeSfcServiceGraph.create_sfc_service_graph()
    columns = ('ID', 'Name', 'Branching Points')
    columns_long = ('Branching Points', 'Description', 'ID', 'Name', 'Project')

    def get_data(self):
        return (self._service_graph['port_chains'], self._service_graph['description'], self._service_graph['id'], self._service_graph['name'], self._service_graph['project_id'])

    def setUp(self):
        super(TestCreateSfcServiceGraph, self).setUp()
        self.network.create_sfc_service_graph = mock.Mock(return_value=self._service_graph)
        self.data = self.get_data()
        self.cmd = sfc_service_graph.CreateSfcServiceGraph(self.app, self.namespace)

    def test_create_sfc_service_graph(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_create_sfc_service_graph_without_loop(self):
        bp1_str = 'pc1:pc2,pc3'
        bp2_str = 'pc2:pc4'
        self.cmd = sfc_service_graph.CreateSfcServiceGraph(self.app, self.namespace)
        arglist = ['--description', self._service_graph['description'], '--branching-point', bp1_str, '--branching-point', bp2_str, self._service_graph['name']]
        pcs = {'pc1': ['pc2', 'pc3'], 'pc2': ['pc4']}
        verifylist = [('description', self._service_graph['description']), ('branching_points', [bp1_str, bp2_str]), ('name', self._service_graph['name'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network.create_sfc_service_graph.assert_called_once_with(**{'description': self._service_graph['description'], 'name': self._service_graph['name'], 'port_chains': pcs})
        self.assertEqual(self.columns_long, columns)
        self.assertEqual(self.data, data)

    def test_create_sfc_service_graph_with_loop(self):
        bp1_str = 'pc1:pc2,pc3;'
        bp2_str = 'pc2:pc1'
        self.cmd = sfc_service_graph.CreateSfcServiceGraph(self.app, self.namespace)
        arglist = ['--description', self._service_graph['description'], '--branching-point', bp1_str, '--branching-point', bp2_str, self._service_graph['name']]
        verifylist = [('description', self._service_graph['description']), ('branching_points', [bp1_str, bp2_str]), ('name', self._service_graph['name'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_create_sfc_service_graph_invalid_port_chains(self):
        bp1_str = 'pc1:pc2,pc3:'
        self.cmd = sfc_service_graph.CreateSfcServiceGraph(self.app, self.namespace)
        arglist = ['--description', self._service_graph['description'], '--branching-point', bp1_str, self._service_graph['name']]
        verifylist = [('description', self._service_graph['description']), ('branching_points', [bp1_str]), ('name', self._service_graph['name'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_create_sfc_service_graph_duplicate_src_chains(self):
        bp1_str = 'pc1:pc2,pc3;'
        bp2_str = 'pc1:pc4'
        self.cmd = sfc_service_graph.CreateSfcServiceGraph(self.app, self.namespace)
        arglist = ['--description', self._service_graph['description'], '--branching-point', bp1_str, '--branching-point', bp2_str, self._service_graph['name']]
        verifylist = [('description', self._service_graph['description']), ('branching_points', [bp1_str, bp2_str]), ('name', self._service_graph['name'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)