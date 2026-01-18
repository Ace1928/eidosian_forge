from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_attachment
def test_volume_attachment_list_with_options(self):
    self.volume_client.api_version = api_versions.APIVersion('3.27')
    arglist = ['--project', self.project.name, '--volume-id', 'volume-id', '--status', 'attached', '--marker', 'volume-attachment-id', '--limit', '2']
    verifylist = [('project', self.project.name), ('all_projects', False), ('volume_id', 'volume-id'), ('status', 'attached'), ('marker', 'volume-attachment-id'), ('limit', 2)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_attachments_mock.list.assert_called_once_with(search_opts={'all_tenants': True, 'project_id': self.project.id, 'status': 'attached', 'volume_id': 'volume-id'}, marker='volume-attachment-id', limit=2)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(tuple(self.data), data)