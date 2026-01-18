from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils as tests_utils
import testtools
from neutronclient.osc.v2.sfc import sfc_service_graph
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_service_graph_show(self):
    client = self.app.client_manager.network
    mock_service_graph_show = client.get_sfc_service_graph
    arglist = [self._service_graph_id]
    verifylist = [('service_graph', self._service_graph_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    mock_service_graph_show.assert_called_once_with(self._service_graph_id)
    self.assertEqual(self.columns_long, columns)
    self.assertEqual(self.data_long, data)