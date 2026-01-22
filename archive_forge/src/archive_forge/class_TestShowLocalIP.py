from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import local_ip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowLocalIP(TestLocalIP):
    _local_ip = network_fakes.create_one_local_ip()
    columns = ('created_at', 'description', 'id', 'name', 'project_id', 'local_port_id', 'network_id', 'local_ip_address', 'ip_mode', 'revision_number', 'updated_at')
    data = (_local_ip.created_at, _local_ip.description, _local_ip.id, _local_ip.name, _local_ip.project_id, _local_ip.local_port_id, _local_ip.network_id, _local_ip.local_ip_address, _local_ip.ip_mode, _local_ip.revision_number, _local_ip.updated_at)

    def setUp(self):
        super().setUp()
        self.network_client.find_local_ip = mock.Mock(return_value=self._local_ip)
        self.cmd = local_ip.ShowLocalIP(self.app, self.namespace)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show_all_options(self):
        arglist = [self._local_ip.name]
        verifylist = [('local_ip', self._local_ip.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_local_ip.assert_called_once_with(self._local_ip.name, ignore_missing=False)
        self.assertEqual(set(self.columns), set(columns))
        self.assertCountEqual(self.data, list(data))