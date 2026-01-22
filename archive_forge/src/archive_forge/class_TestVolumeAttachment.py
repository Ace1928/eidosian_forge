import contextlib
import datetime
from unittest import mock
import uuid
import warnings
from openstack.block_storage.v3 import volume
from openstack.compute.v2 import _proxy
from openstack.compute.v2 import aggregate
from openstack.compute.v2 import availability_zone as az
from openstack.compute.v2 import extension
from openstack.compute.v2 import flavor
from openstack.compute.v2 import hypervisor
from openstack.compute.v2 import image
from openstack.compute.v2 import keypair
from openstack.compute.v2 import migration
from openstack.compute.v2 import quota_set
from openstack.compute.v2 import server
from openstack.compute.v2 import server_action
from openstack.compute.v2 import server_group
from openstack.compute.v2 import server_interface
from openstack.compute.v2 import server_ip
from openstack.compute.v2 import server_migration
from openstack.compute.v2 import server_remote_console
from openstack.compute.v2 import service
from openstack.compute.v2 import usage
from openstack.compute.v2 import volume_attachment
from openstack import resource
from openstack.tests.unit import test_proxy_base
from openstack import warnings as os_warnings
class TestVolumeAttachment(TestComputeProxy):

    def test_volume_attachment_create(self):
        self.verify_create(self.proxy.create_volume_attachment, volume_attachment.VolumeAttachment, method_kwargs={'server': 'server_id', 'volume': 'volume_id'}, expected_kwargs={'server_id': 'server_id', 'volume_id': 'volume_id'})

    def test_volume_attachment_create__legacy_parameters(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.verify_create(self.proxy.create_volume_attachment, volume_attachment.VolumeAttachment, method_kwargs={'server': 'server_id', 'volumeId': 'volume_id'}, expected_kwargs={'server_id': 'server_id', 'volume_id': 'volume_id'})
            self.assertEqual(1, len(w))
            self.assertEqual(os_warnings.OpenStackDeprecationWarning, w[-1].category)
            self.assertIn('This method was called with a volume_id or volumeId argument', str(w[-1]))

    def test_volume_attachment_create__missing_parameters(self):
        exc = self.assertRaises(TypeError, self.proxy.create_volume_attachment, 'server_id')
        self.assertIn('create_volume_attachment() missing 1 required positional argument: volume', str(exc))

    def test_volume_attachment_update(self):
        self.verify_update(self.proxy.update_volume_attachment, volume_attachment.VolumeAttachment, method_args=[], method_kwargs={'server': 'server_id', 'volume': 'volume_id'}, expected_kwargs={'id': 'volume_id', 'server_id': 'server_id', 'volume_id': 'volume_id'})

    def test_volume_attachment_delete(self):
        fake_server = server.Server(id=str(uuid.uuid4()))
        fake_volume = volume.Volume(id=str(uuid.uuid4()))
        self.verify_delete(self.proxy.delete_volume_attachment, volume_attachment.VolumeAttachment, ignore_missing=False, method_args=[fake_server, fake_volume], method_kwargs={}, expected_args=[], expected_kwargs={'id': fake_volume.id, 'server_id': fake_server.id})

    def test_volume_attachment_delete__ignore(self):
        fake_server = server.Server(id=str(uuid.uuid4()))
        fake_volume = volume.Volume(id=str(uuid.uuid4()))
        self.verify_delete(self.proxy.delete_volume_attachment, volume_attachment.VolumeAttachment, ignore_missing=True, method_args=[fake_server, fake_volume], method_kwargs={}, expected_args=[], expected_kwargs={'id': fake_volume.id, 'server_id': fake_server.id})

    def test_volume_attachment_delete__legacy_parameters(self):
        fake_server = server.Server(id=str(uuid.uuid4()))
        fake_volume = volume.Volume(id=str(uuid.uuid4()))
        with mock.patch.object(self.proxy, 'find_server', return_value=None) as mock_find_server:
            self.verify_delete(self.proxy.delete_volume_attachment, volume_attachment.VolumeAttachment, ignore_missing=False, method_args=[fake_volume.id, fake_server.id], method_kwargs={}, expected_args=[], expected_kwargs={'id': fake_volume.id, 'server_id': fake_server.id})
            mock_find_server.assert_called_once_with(fake_volume.id, ignore_missing=True)

    def test_volume_attachment_get(self):
        self.verify_get(self.proxy.get_volume_attachment, volume_attachment.VolumeAttachment, method_args=[], method_kwargs={'server': 'server_id', 'volume': 'volume_id'}, expected_kwargs={'id': 'volume_id', 'server_id': 'server_id'})

    def test_volume_attachments(self):
        self.verify_list(self.proxy.volume_attachments, volume_attachment.VolumeAttachment, method_kwargs={'server': 'server_id'}, expected_kwargs={'server_id': 'server_id'})