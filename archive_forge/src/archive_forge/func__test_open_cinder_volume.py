import contextlib
import hashlib
import io
import math
import os
from unittest import mock
import socket
import sys
import tempfile
import time
import uuid
from keystoneauth1 import exceptions as keystone_exc
from os_brick.initiator import connector
from oslo_concurrency import processutils
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers.cinder import scaleio
from glance_store.common import attachment_state_manager
from glance_store.common import cinder_utils
from glance_store import exceptions
from glance_store import location
from glance_store._drivers.cinder import store as cinder # noqa
from glance_store._drivers.cinder import nfs # noqa
def _test_open_cinder_volume(self, open_mode, attach_mode, error, multipath_supported=False, enforce_multipath=False, encrypted_nfs=False, qcow2_vol=False, multiattach=False, update_attachment_error=None):
    fake_volume = mock.MagicMock(id=str(uuid.uuid4()), status='available', multiattach=multiattach)
    fake_volume.manager.get.return_value = fake_volume
    fake_attachment_id = str(uuid.uuid4())
    fake_attachment_create = {'id': fake_attachment_id}
    if encrypted_nfs or qcow2_vol:
        fake_attachment_update = mock.MagicMock(id=fake_attachment_id, connection_info={'driver_volume_type': 'nfs'})
    else:
        fake_attachment_update = mock.MagicMock(id=fake_attachment_id, connection_info={'driver_volume_type': 'fake'})
    fake_conn_info = mock.MagicMock(connector={})
    fake_volumes = mock.MagicMock(get=lambda id: fake_volume)
    fake_client = mock.MagicMock(volumes=fake_volumes)
    _, fake_dev_path = tempfile.mkstemp(dir=self.test_dir)
    fake_devinfo = {'path': fake_dev_path}
    fake_connector = mock.MagicMock(connect_volume=mock.Mock(return_value=fake_devinfo), disconnect_volume=mock.Mock())

    @contextlib.contextmanager
    def fake_chown(path):
        yield

    def do_open():
        if multiattach:
            with mock.patch.object(attachment_state_manager._AttachmentStateManager, 'get_state') as mock_get_state:
                mock_get_state.return_value.__enter__.return_value = attachment_state_manager._AttachmentState()
                with self.store._open_cinder_volume(fake_client, fake_volume, open_mode):
                    pass
        else:
            with self.store._open_cinder_volume(fake_client, fake_volume, open_mode):
                if error:
                    raise error

    def fake_factory(protocol, root_helper, **kwargs):
        return fake_connector
    root_helper = 'sudo glance-rootwrap /etc/glance/rootwrap.conf'
    with mock.patch.object(cinder.Store, '_wait_volume_status', return_value=fake_volume), mock.patch.object(cinder.Store, 'temporary_chown', side_effect=fake_chown), mock.patch.object(cinder.Store, 'get_root_helper', return_value=root_helper), mock.patch.object(connector.InitiatorConnector, 'factory', side_effect=fake_factory) as fake_conn_obj, mock.patch.object(cinder_utils.API, 'attachment_create', return_value=fake_attachment_create) as attach_create, mock.patch.object(cinder_utils.API, 'attachment_update', return_value=fake_attachment_update) as attach_update, mock.patch.object(cinder_utils.API, 'attachment_delete') as attach_delete, mock.patch.object(cinder_utils.API, 'attachment_get') as attach_get, mock.patch.object(cinder_utils.API, 'attachment_complete') as attach_complete, mock.patch.object(socket, 'gethostname') as mock_get_host, mock.patch.object(socket, 'getaddrinfo') as mock_get_host_ip, mock.patch.object(cinder.strutils, 'mask_dict_password'):
        if update_attachment_error:
            attach_update.side_effect = update_attachment_error
        fake_host = 'fake_host'
        fake_addr_info = [[0, 1, 2, 3, ['127.0.0.1']]]
        fake_ip = fake_addr_info[0][4][0]
        mock_get_host.return_value = fake_host
        mock_get_host_ip.return_value = fake_addr_info
        with mock.patch.object(connector, 'get_connector_properties', return_value=fake_conn_info) as mock_conn:
            if error:
                self.assertRaises(error, do_open)
            elif encrypted_nfs or qcow2_vol:
                fake_volume.encrypted = False
                if encrypted_nfs:
                    fake_volume.encrypted = True
                elif qcow2_vol:
                    attach_get.return_value = mock.MagicMock(connection_info={'format': 'qcow2'})
                try:
                    with self.store._open_cinder_volume(fake_client, fake_volume, open_mode):
                        pass
                except exceptions.BackendException:
                    attach_delete.assert_called_once_with(fake_client, fake_attachment_id)
            elif update_attachment_error:
                self.assertRaises(type(update_attachment_error), do_open)
            else:
                do_open()
            if update_attachment_error:
                attach_delete.assert_called_once_with(fake_client, fake_attachment_id)
            elif not (encrypted_nfs or qcow2_vol):
                mock_conn.assert_called_once_with(root_helper, fake_ip, multipath_supported, enforce_multipath, host=fake_host)
                fake_connector.connect_volume.assert_called_once_with(mock.ANY)
                fake_connector.disconnect_volume.assert_called_once_with(mock.ANY, fake_devinfo, force=True)
                fake_conn_obj.assert_called_once_with(mock.ANY, root_helper, conn=mock.ANY, use_multipath=multipath_supported)
                attach_create.assert_called_once_with(fake_client, fake_volume.id, mode=attach_mode)
                attach_update.assert_called_once_with(fake_client, fake_attachment_id, fake_conn_info, mountpoint='glance_store')
                attach_complete.assert_called_once_with(fake_client, fake_attachment_id)
                attach_delete.assert_called_once_with(fake_client, fake_attachment_id)
            else:
                mock_conn.assert_called_once_with(root_helper, fake_ip, multipath_supported, enforce_multipath, host=fake_host)
                fake_connector.connect_volume.assert_not_called()
                fake_connector.disconnect_volume.assert_not_called()
                attach_create.assert_called_once_with(fake_client, fake_volume.id, mode=attach_mode)
                attach_update.assert_called_once_with(fake_client, fake_attachment_id, fake_conn_info, mountpoint='glance_store')
                attach_delete.assert_called_once_with(fake_client, fake_attachment_id)