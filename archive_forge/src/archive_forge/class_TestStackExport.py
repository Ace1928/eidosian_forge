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
class TestStackExport(TestStack):
    columns = ['stack_name', 'stack_status', 'id']
    data = ['my_stack', 'ABANDONED', '1234']
    response = dict(zip(columns, data))

    def setUp(self):
        super(TestStackExport, self).setUp()
        self.cmd = stack.ExportStack(self.app, None)
        self.stack_client.export.return_value = self.response

    def test_stack_export(self):
        arglist = ['my_stack']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        for column in self.columns:
            self.assertIn(column, columns)
        for datum in self.data:
            self.assertIn(datum, data)

    @mock.patch('heatclient.osc.v1.stack.open', create=True)
    def test_stack_export_output_file(self, mock_open):
        arglist = ['my_stack', '--output-file', 'file.json']
        mock_open.return_value = mock.MagicMock(spec=io.IOBase)
        parsed_args = self.check_parser(self.cmd, arglist, [])
        columns, data = self.cmd.take_action(parsed_args)
        mock_open.assert_called_once_with('file.json', 'w')
        self.assertEqual([], columns)
        self.assertIsNone(data)