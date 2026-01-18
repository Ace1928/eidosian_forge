from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_gateway_options_subnet_ipaddress(self):
    arglist = ['--external-gateway', self._network.id, '--fixed-ip', "subnet='abc',ip-address=10.0.1.1", self._router.id, '--enable-snat']
    verifylist = [('router', self._router.id), ('external_gateway', self._network.id), ('fixed_ip', [{'subnet': "'abc'", 'ip-address': '10.0.1.1'}]), ('enable_snat', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.update_router.assert_called_with(self._router, **{'external_gateway_info': {'network_id': self._network.id, 'external_fixed_ips': [{'subnet_id': self._subnet.id, 'ip_address': '10.0.1.1'}], 'enable_snat': True}})
    self.assertIsNone(result)