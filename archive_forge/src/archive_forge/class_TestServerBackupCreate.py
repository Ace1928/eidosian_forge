from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server_backup
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
class TestServerBackupCreate(TestServerBackup):

    def image_columns(self, image):
        columnlist = ('id', 'name', 'owner', 'protected', 'status', 'tags', 'visibility')
        return columnlist

    def image_data(self, image):
        datalist = (image['id'], image['name'], image['owner_id'], image['is_protected'], 'active', format_columns.ListColumn(image.get('tags')), image['visibility'])
        return datalist

    def setUp(self):
        super(TestServerBackupCreate, self).setUp()
        self.cmd = server_backup.CreateServerBackup(self.app, None)

    def setup_images_mock(self, count, servers=None):
        if servers:
            images = image_fakes.create_images(attrs={'name': servers[0].name, 'status': 'active'}, count=count)
        else:
            images = image_fakes.create_images(attrs={'status': 'active'}, count=count)
        self.image_client.find_image = mock.Mock(side_effect=images)
        return images

    def test_server_backup_defaults(self):
        servers = self.setup_servers_mock(count=1)
        images = self.setup_images_mock(count=1, servers=servers)
        arglist = [servers[0].id]
        verifylist = [('name', None), ('type', None), ('rotate', None), ('wait', False), ('server', servers[0].id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.backup_server.assert_called_with(servers[0].id, servers[0].name, '', 1)
        self.assertEqual(self.image_columns(images[0]), columns)
        self.assertCountEqual(self.image_data(images[0]), data)

    def test_server_backup_create_options(self):
        servers = self.setup_servers_mock(count=1)
        images = self.setup_images_mock(count=1, servers=servers)
        arglist = ['--name', 'image', '--type', 'daily', '--rotate', '2', servers[0].id]
        verifylist = [('name', 'image'), ('type', 'daily'), ('rotate', 2), ('server', servers[0].id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.backup_server.assert_called_with(servers[0].id, 'image', 'daily', 2)
        self.assertEqual(self.image_columns(images[0]), columns)
        self.assertCountEqual(self.image_data(images[0]), data)

    @mock.patch.object(common_utils, 'wait_for_status', return_value=False)
    def test_server_backup_wait_fail(self, mock_wait_for_status):
        servers = self.setup_servers_mock(count=1)
        images = self.setup_images_mock(count=1, servers=servers)
        self.image_client.get_image = mock.Mock(side_effect=images[0])
        arglist = ['--name', 'image', '--type', 'daily', '--wait', servers[0].id]
        verifylist = [('name', 'image'), ('type', 'daily'), ('wait', True), ('server', servers[0].id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.compute_sdk_client.backup_server.assert_called_with(servers[0].id, 'image', 'daily', 1)
        mock_wait_for_status.assert_called_once_with(self.image_client.get_image, images[0].id, callback=mock.ANY)

    @mock.patch.object(common_utils, 'wait_for_status', return_value=True)
    def test_server_backup_wait_ok(self, mock_wait_for_status):
        servers = self.setup_servers_mock(count=1)
        images = self.setup_images_mock(count=1, servers=servers)
        self.image_client.get_image = mock.Mock(side_effect=images[0])
        arglist = ['--name', 'image', '--type', 'daily', '--wait', servers[0].id]
        verifylist = [('name', 'image'), ('type', 'daily'), ('wait', True), ('server', servers[0].id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.compute_sdk_client.backup_server.assert_called_with(servers[0].id, 'image', 'daily', 1)
        mock_wait_for_status.assert_called_once_with(self.image_client.get_image, images[0].id, callback=mock.ANY)
        self.assertEqual(self.image_columns(images[0]), columns)
        self.assertCountEqual(self.image_data(images[0]), data)