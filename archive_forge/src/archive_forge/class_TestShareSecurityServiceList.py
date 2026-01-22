import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import security_services as osc_security_services
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
class TestShareSecurityServiceList(TestShareSecurityService):
    columns = ['ID', 'Name', 'Status', 'Type']

    def setUp(self):
        super(TestShareSecurityServiceList, self).setUp()
        self.share_network = manila_fakes.FakeShareNetwork.create_one_share_network()
        self.share_networks_mock.get.return_value = self.share_network
        self.services_list = manila_fakes.FakeShareSecurityService.create_fake_security_services()
        self.security_services_mock.list.return_value = self.services_list
        self.values = (oscutils.get_dict_properties(i._info, self.columns) for i in self.services_list)
        self.cmd = osc_security_services.ListShareSecurityService(self.app, None)

    def test_share_security_service_list_no_args(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.security_services_mock.list.assert_called_with(search_opts={'all_tenants': False, 'status': None, 'name': None, 'type': None, 'user': None, 'dns_ip': None, 'server': None, 'domain': None, 'offset': None, 'limit': None}, detailed=False)
        self.assertEqual(self.columns, columns)
        self.assertEqual(list(self.values), list(data))

    def test_share_security_service_list(self):
        arglist = ['--share-network', self.share_network.id, '--status', self.services_list[0].status, '--name', self.services_list[0].name, '--type', self.services_list[0].type, '--user', self.services_list[0].user, '--dns-ip', self.services_list[0].dns_ip, '--ou', self.services_list[0].ou, '--server', self.services_list[0].server, '--domain', self.services_list[0].domain, '--default-ad-site', self.services_list[0].default_ad_site, '--limit', '1']
        verifylist = [('share_network', self.share_network.id), ('status', self.services_list[0].status), ('name', self.services_list[0].name), ('type', self.services_list[0].type), ('user', self.services_list[0].user), ('dns_ip', self.services_list[0].dns_ip), ('ou', self.services_list[0].ou), ('server', self.services_list[0].server), ('domain', self.services_list[0].domain), ('default_ad_site', self.services_list[0].default_ad_site), ('limit', 1)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.security_services_mock.list.assert_called_with(search_opts={'all_tenants': False, 'status': self.services_list[0].status, 'name': self.services_list[0].name, 'type': self.services_list[0].type, 'user': self.services_list[0].user, 'dns_ip': self.services_list[0].dns_ip, 'server': self.services_list[0].server, 'domain': self.services_list[0].domain, 'default_ad_site': self.services_list[0].default_ad_site, 'offset': None, 'limit': 1, 'ou': self.services_list[0].ou, 'share_network_id': self.share_network.id}, detailed=False)
        self.assertEqual(self.columns, columns)
        self.assertEqual(list(self.values), list(data))

    def test_share_security_service_list_ou_api_version_exception(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.43')
        arglist = ['--ou', self.services_list[0].ou]
        verifylist = [('ou', self.services_list[0].ou)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_security_service_list_ad_site_api_version_exception(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.75')
        arglist = ['--default-ad-site', self.services_list[0].default_ad_site]
        verifylist = [('default_ad_site', self.services_list[0].default_ad_site)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_share_security_service_list_detail_all_projects(self):
        arglist = ['--all-projects', '--detail']
        verifylist = [('all_projects', True), ('detail', True)]
        columns_detail = self.columns.copy()
        columns_detail.append('Project ID')
        columns_detail.append('Share Networks')
        values_detail = (oscutils.get_dict_properties(i._info, columns_detail) for i in self.services_list)
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.security_services_mock.list.assert_called_with(search_opts={'all_tenants': True, 'status': None, 'name': None, 'type': None, 'user': None, 'dns_ip': None, 'server': None, 'domain': None, 'offset': None, 'limit': None}, detailed=True)
        self.assertEqual(columns_detail, columns)
        self.assertEqual(list(values_detail), list(data))