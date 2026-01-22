from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume
class TestVolumeList(TestVolume):
    project = identity_fakes.FakeProject.create_one_project()
    user = identity_fakes.FakeUser.create_one_user()
    columns = ['ID', 'Name', 'Status', 'Size', 'Attached to']

    def setUp(self):
        super().setUp()
        self.mock_volume = volume_fakes.create_one_volume()
        self.volumes_mock.list.return_value = [self.mock_volume]
        self.users_mock.get.return_value = self.user
        self.projects_mock.get.return_value = self.project
        self.cmd = volume.ListVolume(self.app, None)

    def test_volume_list_no_options(self):
        arglist = []
        verifylist = [('long', False), ('all_projects', False), ('name', None), ('status', None), ('marker', None), ('limit', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        search_opts = {'all_tenants': False, 'project_id': None, 'user_id': None, 'name': None, 'status': None}
        self.volumes_mock.list.assert_called_once_with(search_opts=search_opts, marker=None, limit=None)
        self.assertEqual(self.columns, columns)
        datalist = ((self.mock_volume.id, self.mock_volume.name, self.mock_volume.status, self.mock_volume.size, volume.AttachmentsColumn(self.mock_volume.attachments)),)
        self.assertCountEqual(datalist, tuple(data))

    def test_volume_list_project(self):
        arglist = ['--project', self.project.name]
        verifylist = [('project', self.project.name), ('long', False), ('all_projects', False), ('status', None), ('marker', None), ('limit', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        search_opts = {'all_tenants': True, 'project_id': self.project.id, 'user_id': None, 'name': None, 'status': None}
        self.volumes_mock.list.assert_called_once_with(search_opts=search_opts, marker=None, limit=None)
        self.assertEqual(self.columns, columns)
        datalist = ((self.mock_volume.id, self.mock_volume.name, self.mock_volume.status, self.mock_volume.size, volume.AttachmentsColumn(self.mock_volume.attachments)),)
        self.assertCountEqual(datalist, tuple(data))

    def test_volume_list_project_domain(self):
        arglist = ['--project', self.project.name, '--project-domain', self.project.domain_id]
        verifylist = [('project', self.project.name), ('project_domain', self.project.domain_id), ('long', False), ('all_projects', False), ('status', None), ('marker', None), ('limit', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        search_opts = {'all_tenants': True, 'project_id': self.project.id, 'user_id': None, 'name': None, 'status': None}
        self.volumes_mock.list.assert_called_once_with(search_opts=search_opts, marker=None, limit=None)
        self.assertEqual(self.columns, columns)
        datalist = ((self.mock_volume.id, self.mock_volume.name, self.mock_volume.status, self.mock_volume.size, volume.AttachmentsColumn(self.mock_volume.attachments)),)
        self.assertCountEqual(datalist, tuple(data))

    def test_volume_list_user(self):
        arglist = ['--user', self.user.name]
        verifylist = [('user', self.user.name), ('long', False), ('all_projects', False), ('status', None), ('marker', None), ('limit', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        search_opts = {'all_tenants': False, 'project_id': None, 'user_id': self.user.id, 'name': None, 'status': None}
        self.volumes_mock.list.assert_called_once_with(search_opts=search_opts, marker=None, limit=None)
        self.assertEqual(self.columns, columns)
        datalist = ((self.mock_volume.id, self.mock_volume.name, self.mock_volume.status, self.mock_volume.size, volume.AttachmentsColumn(self.mock_volume.attachments)),)
        self.assertCountEqual(datalist, tuple(data))

    def test_volume_list_user_domain(self):
        arglist = ['--user', self.user.name, '--user-domain', self.user.domain_id]
        verifylist = [('user', self.user.name), ('user_domain', self.user.domain_id), ('long', False), ('all_projects', False), ('status', None), ('marker', None), ('limit', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        search_opts = {'all_tenants': False, 'project_id': None, 'user_id': self.user.id, 'name': None, 'status': None}
        self.volumes_mock.list.assert_called_once_with(search_opts=search_opts, marker=None, limit=None)
        self.assertEqual(self.columns, columns)
        datalist = ((self.mock_volume.id, self.mock_volume.name, self.mock_volume.status, self.mock_volume.size, volume.AttachmentsColumn(self.mock_volume.attachments)),)
        self.assertCountEqual(datalist, tuple(data))

    def test_volume_list_name(self):
        arglist = ['--name', self.mock_volume.name]
        verifylist = [('long', False), ('all_projects', False), ('name', self.mock_volume.name), ('status', None), ('marker', None), ('limit', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        search_opts = {'all_tenants': False, 'project_id': None, 'user_id': None, 'name': self.mock_volume.name, 'status': None}
        self.volumes_mock.list.assert_called_once_with(search_opts=search_opts, marker=None, limit=None)
        self.assertEqual(self.columns, columns)
        datalist = ((self.mock_volume.id, self.mock_volume.name, self.mock_volume.status, self.mock_volume.size, volume.AttachmentsColumn(self.mock_volume.attachments)),)
        self.assertCountEqual(datalist, tuple(data))

    def test_volume_list_status(self):
        arglist = ['--status', self.mock_volume.status]
        verifylist = [('long', False), ('all_projects', False), ('name', None), ('status', self.mock_volume.status), ('marker', None), ('limit', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        search_opts = {'all_tenants': False, 'project_id': None, 'user_id': None, 'name': None, 'status': self.mock_volume.status}
        self.volumes_mock.list.assert_called_once_with(search_opts=search_opts, marker=None, limit=None)
        self.assertEqual(self.columns, columns)
        datalist = ((self.mock_volume.id, self.mock_volume.name, self.mock_volume.status, self.mock_volume.size, volume.AttachmentsColumn(self.mock_volume.attachments)),)
        self.assertCountEqual(datalist, tuple(data))

    def test_volume_list_all_projects(self):
        arglist = ['--all-projects']
        verifylist = [('long', False), ('all_projects', True), ('name', None), ('status', None), ('marker', None), ('limit', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        search_opts = {'all_tenants': True, 'project_id': None, 'user_id': None, 'name': None, 'status': None}
        self.volumes_mock.list.assert_called_once_with(search_opts=search_opts, marker=None, limit=None)
        self.assertEqual(self.columns, columns)
        datalist = ((self.mock_volume.id, self.mock_volume.name, self.mock_volume.status, self.mock_volume.size, volume.AttachmentsColumn(self.mock_volume.attachments)),)
        self.assertCountEqual(datalist, tuple(data))

    def test_volume_list_long(self):
        arglist = ['--long']
        verifylist = [('long', True), ('all_projects', False), ('name', None), ('status', None), ('marker', None), ('limit', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        search_opts = {'all_tenants': False, 'project_id': None, 'user_id': None, 'name': None, 'status': None}
        self.volumes_mock.list.assert_called_once_with(search_opts=search_opts, marker=None, limit=None)
        collist = ['ID', 'Name', 'Status', 'Size', 'Type', 'Bootable', 'Attached to', 'Properties']
        self.assertEqual(collist, columns)
        datalist = ((self.mock_volume.id, self.mock_volume.name, self.mock_volume.status, self.mock_volume.size, self.mock_volume.volume_type, self.mock_volume.bootable, volume.AttachmentsColumn(self.mock_volume.attachments), format_columns.DictColumn(self.mock_volume.metadata)),)
        self.assertCountEqual(datalist, tuple(data))

    def test_volume_list_with_marker_and_limit(self):
        arglist = ['--marker', self.mock_volume.id, '--limit', '2']
        verifylist = [('long', False), ('all_projects', False), ('name', None), ('status', None), ('marker', self.mock_volume.id), ('limit', 2)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        datalist = ((self.mock_volume.id, self.mock_volume.name, self.mock_volume.status, self.mock_volume.size, volume.AttachmentsColumn(self.mock_volume.attachments)),)
        self.volumes_mock.list.assert_called_once_with(marker=self.mock_volume.id, limit=2, search_opts={'status': None, 'project_id': None, 'user_id': None, 'name': None, 'all_tenants': False})
        self.assertCountEqual(datalist, tuple(data))

    def test_volume_list_negative_limit(self):
        arglist = ['--limit', '-2']
        verifylist = [('limit', -2)]
        self.assertRaises(test_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_volume_list_backward_compatibility(self):
        arglist = ['-c', 'Display Name']
        verifylist = [('columns', ['Display Name']), ('long', False), ('all_projects', False), ('name', None), ('status', None), ('marker', None), ('limit', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        search_opts = {'all_tenants': False, 'project_id': None, 'user_id': None, 'name': None, 'status': None}
        self.volumes_mock.list.assert_called_once_with(search_opts=search_opts, marker=None, limit=None)
        self.assertIn('Display Name', columns)
        self.assertNotIn('Name', columns)
        for each_volume in data:
            self.assertIn(self.mock_volume.name, each_volume)