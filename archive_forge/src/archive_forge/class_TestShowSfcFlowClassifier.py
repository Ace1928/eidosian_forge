from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_flow_classifier
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestShowSfcFlowClassifier(fakes.TestNeutronClientOSCV2):
    _fc = fakes.FakeSfcFlowClassifier.create_flow_classifier()
    data = (_fc['description'], _fc['destination_ip_prefix'], _fc['destination_port_range_max'], _fc['destination_port_range_min'], _fc['ethertype'], _fc['id'], _fc['l7_parameters'], _fc['logical_destination_port'], _fc['logical_source_port'], _fc['name'], _fc['project_id'], _fc['protocol'], _fc['source_ip_prefix'], _fc['source_port_range_max'], _fc['source_port_range_min'])
    _flow_classifier = _fc
    _flow_classifier_id = _fc['id']
    columns = ('Description', 'Destination IP', 'Destination Port Range Max', 'Destination Port Range Min', 'Ethertype', 'ID', 'L7 Parameters', 'Logical Destination Port', 'Logical Source Port', 'Name', 'Project', 'Protocol', 'Source IP', 'Source Port Range Max', 'Source Port Range Min', 'Summary')

    def setUp(self):
        super(TestShowSfcFlowClassifier, self).setUp()
        self.network.get_sfc_flow_classifier = mock.Mock(return_value=self._flow_classifier)
        self.cmd = sfc_flow_classifier.ShowSfcFlowClassifier(self.app, self.namespace)

    def test_show_flow_classifier(self):
        client = self.app.client_manager.network
        mock_flow_classifier_show = client.get_sfc_flow_classifier
        arglist = [self._flow_classifier_id]
        verifylist = [('flow_classifier', self._flow_classifier_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        mock_flow_classifier_show.assert_called_once_with(self._flow_classifier_id)
        self.assertEqual(self.columns, columns)