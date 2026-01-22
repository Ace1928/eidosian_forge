from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet as subnet_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def _test_create_with_dns(self, publish_dns=True):
    arglist = ['--subnet-range', self._subnet.cidr, '--network', self._subnet.network_id, self._subnet.name]
    if publish_dns:
        arglist += ['--dns-publish-fixed-ip']
    else:
        arglist += ['--no-dns-publish-fixed-ip']
    verifylist = [('name', self._subnet.name), ('subnet_range', self._subnet.cidr), ('network', self._subnet.network_id), ('ip_version', self._subnet.ip_version), ('gateway', 'auto')]
    verifylist.append(('dns_publish_fixed_ip', publish_dns))
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_subnet.assert_called_once_with(cidr=self._subnet.cidr, ip_version=self._subnet.ip_version, name=self._subnet.name, network_id=self._subnet.network_id, dns_publish_fixed_ip=publish_dns)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)