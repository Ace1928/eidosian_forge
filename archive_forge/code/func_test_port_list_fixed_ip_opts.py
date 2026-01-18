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
def test_port_list_fixed_ip_opts(self):
    subnet_id = self._ports[0].fixed_ips[0]['subnet_id']
    ip_address = self._ports[0].fixed_ips[0]['ip_address']
    arglist = ['--fixed-ip', 'subnet=%s,ip-address=%s' % (subnet_id, ip_address)]
    verifylist = [('fixed_ip', [{'subnet': subnet_id, 'ip-address': ip_address}])]
    self.fake_subnet = network_fakes.FakeSubnet.create_one_subnet({'id': subnet_id})
    self.network_client.find_subnet = mock.Mock(return_value=self.fake_subnet)
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.ports.assert_called_once_with(**{'fixed_ips': ['subnet_id=%s' % subnet_id, 'ip_address=%s' % ip_address], 'fields': LIST_FIELDS_TO_RETRIEVE})
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))