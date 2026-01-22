from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_flow_classifier
from neutronclient.tests.unit.osc.v2.sfc import fakes
class TestListSfcFlowClassifier(fakes.TestNeutronClientOSCV2):
    _fc = fakes.FakeSfcFlowClassifier.create_flow_classifiers(count=1)
    columns = ('ID', 'Name', 'Summary')
    columns_long = ('ID', 'Name', 'Protocol', 'Ethertype', 'Source IP', 'Destination IP', 'Logical Source Port', 'Logical Destination Port', 'Source Port Range Min', 'Source Port Range Max', 'Destination Port Range Min', 'Destination Port Range Max', 'L7 Parameters', 'Description', 'Project')
    _flow_classifier = _fc[0]
    data = [_flow_classifier['id'], _flow_classifier['name'], _flow_classifier['protocol'], _flow_classifier['source_ip_prefix'], _flow_classifier['destination_ip_prefix'], _flow_classifier['logical_source_port'], _flow_classifier['logical_destination_port']]
    data_long = [_flow_classifier['id'], _flow_classifier['name'], _flow_classifier['protocol'], _flow_classifier['ethertype'], _flow_classifier['source_ip_prefix'], _flow_classifier['destination_ip_prefix'], _flow_classifier['logical_source_port'], _flow_classifier['logical_destination_port'], _flow_classifier['source_port_range_min'], _flow_classifier['source_port_range_max'], _flow_classifier['destination_port_range_min'], _flow_classifier['destination_port_range_max'], _flow_classifier['l7_parameters'], _flow_classifier['description']]
    _flow_classifier1 = {'flow_classifiers': _flow_classifier}
    _flow_classifier_id = _flow_classifier['id']

    def setUp(self):
        super(TestListSfcFlowClassifier, self).setUp()
        self.network.sfc_flow_classifiers = mock.Mock(return_value=self._fc)
        self.cmd = sfc_flow_classifier.ListSfcFlowClassifier(self.app, self.namespace)

    def test_list_flow_classifiers(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns = self.cmd.take_action(parsed_args)
        fcs = self.network.sfc_flow_classifiers()
        fc = fcs[0]
        data = [fc['id'], fc['name'], fc['protocol'], fc['source_ip_prefix'], fc['destination_ip_prefix'], fc['logical_source_port'], fc['logical_destination_port']]
        self.assertEqual(list(self.columns), columns[0])
        self.assertEqual(self.data, data)

    def test_list_with_long_option(self):
        arglist = ['--long']
        verifylist = [('long', True)]
        fcs = self.network.sfc_flow_classifiers()
        fc = fcs[0]
        data = [fc['id'], fc['name'], fc['protocol'], fc['ethertype'], fc['source_ip_prefix'], fc['destination_ip_prefix'], fc['logical_source_port'], fc['logical_destination_port'], fc['source_port_range_min'], fc['source_port_range_max'], fc['destination_port_range_min'], fc['destination_port_range_max'], fc['l7_parameters'], fc['description']]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns_long = self.cmd.take_action(parsed_args)[0]
        self.assertEqual(list(self.columns_long), columns_long)
        self.assertEqual(self.data_long, data)