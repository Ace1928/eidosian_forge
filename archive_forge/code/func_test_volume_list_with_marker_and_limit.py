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
def test_volume_list_with_marker_and_limit(self):
    arglist = ['--marker', self.mock_volume.id, '--limit', '2']
    verifylist = [('long', False), ('all_projects', False), ('name', None), ('status', None), ('marker', self.mock_volume.id), ('limit', 2)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    datalist = ((self.mock_volume.id, self.mock_volume.name, self.mock_volume.status, self.mock_volume.size, volume.AttachmentsColumn(self.mock_volume.attachments)),)
    self.volumes_mock.list.assert_called_once_with(marker=self.mock_volume.id, limit=2, search_opts={'status': None, 'project_id': None, 'user_id': None, 'name': None, 'all_tenants': False})
    self.assertCountEqual(datalist, tuple(data))