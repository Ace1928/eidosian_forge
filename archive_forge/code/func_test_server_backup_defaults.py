from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server_backup
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
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