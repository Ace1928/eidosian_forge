import hashlib
import logging
import os
import socket
from oslo_config import cfg
from glance_store._drivers.cinder import base
from glance_store.common import cinder_utils
from glance_store.common import fs_mount as mount
from glance_store.common import utils
from glance_store import exceptions
from glance_store.i18n import _
class NfsBrickConnector(base.BaseBrickConnectorInterface):

    def __init__(self, *args, **kwargs):
        self.volume = kwargs.get('volume')
        self.connection_info = kwargs.get('connection_info')
        self.root_helper = kwargs.get('root_helper')
        self.mount_point_base = kwargs.get('mountpoint_base')
        self.attachment_obj = kwargs.get('attachment_obj')
        self.client = kwargs.get('client')
        self.host = socket.gethostname()
        self.volume_api = cinder_utils.API()

    def _get_mount_path(self, share, mount_point_base):
        """Returns the mount path prefix using the mount point base and share.

        :returns: The mount path prefix.
        """
        return os.path.join(self.mount_point_base, NfsBrickConnector.get_hash_str(share))

    @staticmethod
    def get_hash_str(base_str):
        """Returns string representing SHA256 hash of base_str in hex format.

        If base_str is a Unicode string, encode it to UTF-8.
        """
        if isinstance(base_str, str):
            base_str = base_str.encode('utf-8')
        return hashlib.sha256(base_str).hexdigest()

    def connect_volume(self, volume):
        vol_attachment = self.volume_api.attachment_get(self.client, self.attachment_obj.id)
        if volume.encrypted or vol_attachment.connection_info['format'] == 'qcow2':
            issue_type = 'Encrypted' if volume.encrypted else 'qcow2'
            msg = _('%(issue_type)s volume creation for cinder nfs is not supported from glance_store. Failed to create volume %(volume_id)s') % {'issue_type': issue_type, 'volume_id': volume.id}
            LOG.error(msg)
            raise exceptions.BackendException(msg)

        @utils.synchronized(self.connection_info['export'])
        def connect_volume_nfs():
            export = self.connection_info['export']
            vol_name = self.connection_info['name']
            mountpoint = self._get_mount_path(export, os.path.join(self.mount_point_base, 'nfs'))
            options = self.connection_info['options']
            mount.mount('nfs', export, vol_name, mountpoint, self.host, self.root_helper, options)
            return {'path': os.path.join(mountpoint, vol_name)}
        device = connect_volume_nfs()
        return device

    def disconnect_volume(self, device):

        @utils.synchronized(self.connection_info['export'])
        def disconnect_volume_nfs():
            path, vol_name = device['path'].rsplit('/', 1)
            mount.umount(vol_name, path, self.host, self.root_helper)
        disconnect_volume_nfs()

    def extend_volume(self):
        raise NotImplementedError