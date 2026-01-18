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