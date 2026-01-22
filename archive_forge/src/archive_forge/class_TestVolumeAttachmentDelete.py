from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_attachment
class TestVolumeAttachmentDelete(TestVolumeAttachment):
    volume_attachment = volume_fakes.create_one_volume_attachment()

    def setUp(self):
        super().setUp()
        self.volume_attachments_mock.delete.return_value = None
        self.cmd = volume_attachment.DeleteVolumeAttachment(self.app, None)

    def test_volume_attachment_delete(self):
        self.volume_client.api_version = api_versions.APIVersion('3.27')
        arglist = [self.volume_attachment.id]
        verifylist = [('attachment', self.volume_attachment.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.volume_attachments_mock.delete.assert_called_once_with(self.volume_attachment.id)
        self.assertIsNone(result)

    def test_volume_attachment_delete_pre_v327(self):
        self.volume_client.api_version = api_versions.APIVersion('3.26')
        arglist = [self.volume_attachment.id]
        verifylist = [('attachment', self.volume_attachment.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.27 or greater is required', str(exc))