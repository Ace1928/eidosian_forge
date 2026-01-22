import copy
import io
from unittest import mock
from osc_lib import exceptions as exc
from osc_lib import utils
import testscenarios
import yaml
from heatclient.common import template_format
from heatclient import exc as heat_exc
from heatclient.osc.v1 import stack
from heatclient.tests import inline_templates
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import events
from heatclient.v1 import resources
from heatclient.v1 import stacks
class TestStackAbandon(TestStack):
    columns = ['stack_name', 'stack_status', 'id']
    data = ['my_stack', 'ABANDONED', '1234']
    response = dict(zip(columns, data))

    def setUp(self):
        super(TestStackAbandon, self).setUp()
        self.cmd = stack.AbandonStack(self.app, None)
        self.stack_client.abandon.return_value = self.response

    def test_stack_abandon(self):
        arglist = ['my_stack']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        for column in self.columns:
            self.assertIn(column, columns)
        for datum in self.data:
            self.assertIn(datum, data)

    def test_stack_abandon_not_found(self):
        arglist = ['my_stack']
        self.stack_client.abandon.side_effect = heat_exc.HTTPNotFound
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)

    @mock.patch('heatclient.osc.v1.stack.open', create=True)
    def test_stack_abandon_output_file(self, mock_open):
        arglist = ['my_stack', '--output-file', 'file.json']
        mock_open.return_value = mock.MagicMock(spec=io.IOBase)
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        mock_open.assert_called_once_with('file.json', 'w')
        self.assertEqual([], columns)
        self.assertIsNone(data)

    @mock.patch('heatclient.osc.v1.stack.open', create=True, side_effect=IOError)
    def test_stack_abandon_output_file_error(self, mock_open):
        arglist = ['my_stack', '--output-file', 'file.json']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)