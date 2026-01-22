from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_attachment
class TestVolumeAttachmentSet(TestVolumeAttachment):
    volume_attachment = volume_fakes.create_one_volume_attachment()
    columns = ('ID', 'Volume ID', 'Instance ID', 'Status', 'Attach Mode', 'Attached At', 'Detached At', 'Properties')
    data = (volume_attachment.id, volume_attachment.volume_id, volume_attachment.instance, volume_attachment.status, volume_attachment.attach_mode, volume_attachment.attached_at, volume_attachment.detached_at, format_columns.DictColumn(volume_attachment.connection_info))

    def setUp(self):
        super().setUp()
        self.volume_attachments_mock.update.return_value = self.volume_attachment
        self.cmd = volume_attachment.SetVolumeAttachment(self.app, None)

    def test_volume_attachment_set(self):
        self.volume_client.api_version = api_versions.APIVersion('3.27')
        arglist = [self.volume_attachment.id, '--initiator', 'iqn.1993-08.org.debian:01:cad181614cec', '--ip', '192.168.1.20', '--host', 'my-host', '--platform', 'x86_64', '--os-type', 'linux2', '--multipath', '--mountpoint', '/dev/vdb']
        verifylist = [('attachment', self.volume_attachment.id), ('initiator', 'iqn.1993-08.org.debian:01:cad181614cec'), ('ip', '192.168.1.20'), ('host', 'my-host'), ('platform', 'x86_64'), ('os_type', 'linux2'), ('multipath', True), ('mountpoint', '/dev/vdb')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        connect_info = dict([('initiator', 'iqn.1993-08.org.debian:01:cad181614cec'), ('ip', '192.168.1.20'), ('host', 'my-host'), ('platform', 'x86_64'), ('os_type', 'linux2'), ('multipath', True), ('mountpoint', '/dev/vdb')])
        self.volume_attachments_mock.update.assert_called_once_with(self.volume_attachment.id, connect_info)
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_volume_attachment_set_pre_v327(self):
        self.volume_client.api_version = api_versions.APIVersion('3.26')
        arglist = [self.volume_attachment.id, '--initiator', 'iqn.1993-08.org.debian:01:cad181614cec']
        verifylist = [('attachment', self.volume_attachment.id), ('initiator', 'iqn.1993-08.org.debian:01:cad181614cec')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.27 or greater is required', str(exc))