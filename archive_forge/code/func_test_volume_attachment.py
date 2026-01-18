from openstack.block_storage.v3 import volume as volume_
from openstack.compute.v2 import server as server_
from openstack.compute.v2 import volume_attachment as volume_attachment_
from openstack.tests.functional.compute import base as ft_base
def test_volume_attachment(self):
    volume_attachment = self.user_cloud.compute.create_volume_attachment(self.server, self.volume)
    self.assertIsInstance(volume_attachment, volume_attachment_.VolumeAttachment)
    self.user_cloud.block_storage.wait_for_status(self.volume, status='in-use', wait=self._wait_for_timeout)
    volume_attachments = list(self.user_cloud.compute.volume_attachments(self.server))
    self.assertEqual(1, len(volume_attachments))
    self.assertIsInstance(volume_attachments[0], volume_attachment_.VolumeAttachment)
    volume_attachment = self.user_cloud.compute.update_volume_attachment(self.server, self.volume, delete_on_termination=True)
    self.assertIsInstance(volume_attachment, volume_attachment_.VolumeAttachment)
    volume_attachment = self.user_cloud.compute.get_volume_attachment(self.server, self.volume)
    self.assertIsInstance(volume_attachment, volume_attachment_.VolumeAttachment)
    self.assertTrue(volume_attachment.delete_on_termination)
    result = self.user_cloud.compute.delete_volume_attachment(self.server, self.volume, ignore_missing=False)
    self.assertIsNone(result)
    self.user_cloud.block_storage.wait_for_status(self.volume, status='available', wait=self._wait_for_timeout)