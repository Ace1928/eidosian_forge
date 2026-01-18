import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import security_services as osc_security_services
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_security_service_list_no_args(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.security_services_mock.list.assert_called_with(search_opts={'all_tenants': False, 'status': None, 'name': None, 'type': None, 'user': None, 'dns_ip': None, 'server': None, 'domain': None, 'offset': None, 'limit': None}, detailed=False)
    self.assertEqual(self.columns, columns)
    self.assertEqual(list(self.values), list(data))