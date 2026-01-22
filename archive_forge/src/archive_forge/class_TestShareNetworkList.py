from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_networks as osc_share_networks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.ddt
class TestShareNetworkList(TestShareNetwork):

    def setUp(self):
        super(TestShareNetworkList, self).setUp()
        self.share_networks = manila_fakes.FakeShareNetwork.create_share_networks(count=2)
        self.share_networks_list = oscutils.sort_items(self.share_networks, 'name:asc', str)
        self.share_networks_mock.list.return_value = self.share_networks_list
        self.values = (oscutils.get_dict_properties(s._info, COLUMNS) for s in self.share_networks_list)
        self.expected_search_opts = {'all_tenants': False, 'project_id': None, 'name': None, 'created_since': None, 'created_before': None, 'neutron_net_id': None, 'neutron_subnet_id': None, 'network_type': None, 'segmentation_id': None, 'cidr': None, 'ip_version': None, 'security_service': None, 'name~': None, 'description~': None, 'description': None}
        self.cmd = osc_share_networks.ListShareNetwork(self.app, None)

    @ddt.data(True, False)
    def test_list_share_networks_with_search_opts(self, with_search_opts):
        if with_search_opts:
            arglist = ['--name', 'foo', '--ip-version', '4', '--description~', 'foo-share-network']
            verifylist = [('name', 'foo'), ('ip_version', '4'), ('description~', 'foo-share-network')]
            self.expected_search_opts.update({'name': 'foo', 'ip_version': '4', 'description~': 'foo-share-network'})
        else:
            arglist = []
            verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.share_networks_mock.list.assert_called_once_with(search_opts=self.expected_search_opts)
        self.assertEqual(COLUMNS, columns)
        self.assertEqual(list(self.values), list(data))

    def test_list_share_networks_all_projects(self):
        all_tenants_list = COLUMNS.copy()
        all_tenants_list.append('Project ID')
        self.expected_search_opts.update({'all_tenants': True})
        list_values = (oscutils.get_dict_properties(s._info, all_tenants_list) for s in self.share_networks_list)
        arglist = ['--all-projects']
        verifylist = [('all_projects', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.share_networks_mock.list.assert_called_once_with(search_opts=self.expected_search_opts)
        self.assertEqual(all_tenants_list, columns)
        self.assertEqual(list(list_values), list(data))

    def test_list_share_networks_detail(self):
        values = (oscutils.get_dict_properties(s._info, COLUMNS_DETAIL) for s in self.share_networks_list)
        arglist = ['--detail']
        verifylist = [('detail', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.share_networks_mock.list.assert_called_once_with(search_opts=self.expected_search_opts)
        self.assertEqual(COLUMNS_DETAIL, columns)
        self.assertEqual(list(values), list(data))

    def test_list_share_networks_api_version_exception(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.35')
        arglist = ['--description', 'Description']
        verifylist = [('description', 'Description')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)