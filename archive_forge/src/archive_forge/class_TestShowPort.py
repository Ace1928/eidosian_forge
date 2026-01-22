from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.network.v2 import port
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
class TestShowPort(TestPort):
    _port = network_fakes.create_one_port()
    columns, data = TestPort._get_common_cols_data(_port)

    def setUp(self):
        super(TestShowPort, self).setUp()
        self.network_client.find_port = mock.Mock(return_value=self._port)
        self.cmd = port.ShowPort(self.app, self.namespace)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show_all_options(self):
        arglist = [self._port.name]
        verifylist = [('port', self._port.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_port.assert_called_once_with(self._port.name, ignore_missing=False)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)