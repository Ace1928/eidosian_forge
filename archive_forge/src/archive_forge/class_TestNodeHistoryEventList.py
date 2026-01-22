import copy
import io
import json
import sys
from unittest import mock
from osc_lib.tests import utils as oscutils
from ironicclient.common import utils as commonutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_node
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
from ironicclient.v1 import utils as v1_utils
class TestNodeHistoryEventList(TestBaremetal):

    def setUp(self):
        super(TestNodeHistoryEventList, self).setUp()
        self.baremetal_mock.node.get_history_list.return_value = baremetal_fakes.NODE_HISTORY
        self.cmd = baremetal_node.NodeHistoryList(self.app, None)

    def test_baremetal_node_history_list(self):
        arglist = ['node_uuid', '--long']
        verifylist = [('node', 'node_uuid'), ('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.baremetal_mock.node.get_history_list.assert_called_once_with('node_uuid', True)
        expected_columns = ('UUID', 'Created At', 'Severity', 'Event Origin Type', 'Description of the event', 'Conductor', 'User')
        expected_data = (('abcdef1', 'time', 'info', 'purring', 'meow', 'lap-conductor', '0191'),)
        self.assertEqual(expected_columns, columns)
        self.assertEqual(expected_data, tuple(data))