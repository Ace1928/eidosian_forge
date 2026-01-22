from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip as fip
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch('openstackclient.api.compute_v2.APIv2.floating_ip_create')
class TestCreateFloatingIPCompute(compute_fakes.TestComputev2):
    _floating_ip = compute_fakes.create_one_floating_ip()
    columns = ('fixed_ip', 'id', 'instance_id', 'ip', 'pool')
    data = (_floating_ip['fixed_ip'], _floating_ip['id'], _floating_ip['instance_id'], _floating_ip['ip'], _floating_ip['pool'])

    def setUp(self):
        super(TestCreateFloatingIPCompute, self).setUp()
        self.app.client_manager.network_endpoint_enabled = False
        self.cmd = fip.CreateFloatingIP(self.app, None)

    def test_floating_ip_create_no_arg(self, fip_mock):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_floating_ip_create_default(self, fip_mock):
        fip_mock.return_value = self._floating_ip
        arglist = [self._floating_ip['pool']]
        verifylist = [('network', self._floating_ip['pool'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        fip_mock.assert_called_once_with(self._floating_ip['pool'])
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, data)