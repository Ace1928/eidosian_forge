from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_flow_classifier
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_show_flow_classifier(self):
    client = self.app.client_manager.network
    mock_flow_classifier_show = client.get_sfc_flow_classifier
    arglist = [self._flow_classifier_id]
    verifylist = [('flow_classifier', self._flow_classifier_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    mock_flow_classifier_show.assert_called_once_with(self._flow_classifier_id)
    self.assertEqual(self.columns, columns)