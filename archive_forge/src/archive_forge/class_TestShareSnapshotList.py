from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share_snapshots as osc_share_snapshots
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.ddt
class TestShareSnapshotList(TestShareSnapshot):

    def setUp(self):
        super(TestShareSnapshotList, self).setUp()
        self.share_snapshots = manila_fakes.FakeShareSnapshot.create_share_snapshots(count=2)
        self.snapshots_list = oscutils.sort_items(self.share_snapshots, 'name:asc', str)
        self.snapshots_mock.list.return_value = self.snapshots_list
        self.values = (oscutils.get_dict_properties(s._info, COLUMNS) for s in self.snapshots_list)
        self.cmd = osc_share_snapshots.ListShareSnapshot(self.app, None)

    def test_list_snapshots(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.list.assert_called_with(search_opts={'offset': None, 'limit': None, 'all_tenants': False, 'name': None, 'status': None, 'share_id': None, 'usage': None, 'metadata': {}, 'name~': None, 'description~': None, 'description': None})
        self.assertEqual(COLUMNS, columns)
        self.assertEqual(list(self.values), list(data))

    def test_list_snapshots_all_projects(self):
        all_tenants_list = COLUMNS.copy()
        all_tenants_list.append('Project ID')
        list_values = (oscutils.get_dict_properties(s._info, all_tenants_list) for s in self.snapshots_list)
        arglist = ['--all-projects']
        verifylist = [('all_projects', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.list.assert_called_with(search_opts={'offset': None, 'limit': None, 'all_tenants': True, 'name': None, 'status': None, 'share_id': None, 'usage': None, 'metadata': {}, 'name~': None, 'description~': None, 'description': None})
        self.assertEqual(all_tenants_list, columns)
        self.assertEqual(list(list_values), list(data))

    def test_list_snapshots_detail(self):
        values = (oscutils.get_dict_properties(s._info, COLUMNS_DETAIL) for s in self.snapshots_list)
        arglist = ['--detail']
        verifylist = [('detail', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.list.assert_called_with(search_opts={'offset': None, 'limit': None, 'all_tenants': False, 'name': None, 'status': None, 'share_id': None, 'usage': None, 'metadata': {}, 'name~': None, 'description~': None, 'description': None})
        self.assertEqual(COLUMNS_DETAIL, columns)
        self.assertEqual(list(values), list(data))

    @ddt.data('2.35', '2.78')
    def test_list_snapshots_api_version_exception(self, v):
        self.app.client_manager.share.api_version = api_versions.APIVersion(v)
        if v == '2.35':
            arglist = ['--description', 'Description']
            verifylist = [('description', 'Description')]
        elif v == '2.78':
            arglist = ['--count']
            verifylist = [('count', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_list_snapshots_share_id(self):
        self.share = manila_fakes.FakeShare.create_one_share(attrs={'id': self.snapshots_list[0].id})
        self.shares_mock.get.return_value = self.share
        self.snapshots_mock.list.return_value = [self.snapshots_list[0]]
        values = (oscutils.get_dict_properties(s._info, COLUMNS) for s in [self.snapshots_list[0]])
        arglist = ['--share', self.share.id]
        verifylist = [('share', self.share.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.list.assert_called_with(search_opts={'offset': None, 'limit': None, 'all_tenants': False, 'name': None, 'status': None, 'share_id': self.share.id, 'usage': None, 'metadata': {}, 'name~': None, 'description~': None, 'description': None})
        self.assertEqual(COLUMNS, columns)
        self.assertEqual(list(values), list(data))

    def test_list_snapshots_with_count(self):
        self.app.client_manager.share.api_version = api_versions.APIVersion('2.79')
        self.snapshots_mock.list.return_value = (self.snapshots_list, 2)
        arglist = ['--count']
        verifylist = [('count', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.snapshots_mock.list.assert_called_with(search_opts={'offset': None, 'limit': None, 'all_tenants': False, 'name': None, 'status': None, 'share_id': None, 'usage': None, 'metadata': {}, 'name~': None, 'description~': None, 'description': None, 'with_count': True})