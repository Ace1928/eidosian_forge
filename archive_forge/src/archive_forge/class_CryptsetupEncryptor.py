import binascii
import os
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_log import versionutils
from os_brick.encryptors import base
from os_brick import exception
class CryptsetupEncryptor(base.VolumeEncryptor):
    """A VolumeEncryptor based on dm-crypt.

    This VolumeEncryptor uses dm-crypt to encrypt the specified volume.
    """

    def __init__(self, root_helper, connection_info, keymgr, execute=None, *args, **kwargs):
        super(CryptsetupEncryptor, self).__init__(*args, root_helper=root_helper, connection_info=connection_info, keymgr=keymgr, execute=execute, **kwargs)
        data = connection_info['data']
        if not data.get('device_path'):
            volume_id = data.get('volume_id') or connection_info.get('serial')
            raise exception.VolumeEncryptionNotSupported(volume_id=volume_id, volume_type=connection_info['driver_volume_type'])
        self.symlink_path = connection_info['data']['device_path']
        self.dev_name = 'crypt-%s' % os.path.basename(self.symlink_path)
        old_dev_name = os.path.basename(self.symlink_path)
        wwn = data.get('multipath_id')
        if self._is_crypt_device_available(old_dev_name):
            self.dev_name = old_dev_name
            LOG.debug('Using old encrypted volume name: %s', self.dev_name)
        elif wwn and wwn != old_dev_name:
            if self._is_crypt_device_available(wwn):
                self.dev_name = wwn
                LOG.debug('Using encrypted volume name from wwn: %s', self.dev_name)
        self.dev_path = os.path.realpath(self.symlink_path)

    def _is_crypt_device_available(self, dev_name):
        if not os.path.exists('/dev/mapper/%s' % dev_name):
            return False
        try:
            self._execute('cryptsetup', 'status', dev_name, run_as_root=True)
        except processutils.ProcessExecutionError as e:
            if e.exit_code != 1:
                LOG.warning('cryptsetup status %(dev_name)s exited abnormally (status %(exit_code)s): %(err)s', {'dev_name': dev_name, 'exit_code': e.exit_code, 'err': e.stderr})
            return False
        return True

    def _get_passphrase(self, key):
        """Convert raw key to string."""
        return binascii.hexlify(key).decode('utf-8')

    def _open_volume(self, passphrase, **kwargs):
        """Open the LUKS partition on the volume using passphrase.

        :param passphrase: the passphrase used to access the volume
        """
        LOG.debug('opening encrypted volume %s', self.dev_path)
        cmd = ['cryptsetup', 'create', '--key-file=-']
        cipher = kwargs.get('cipher', None)
        if cipher is not None:
            cmd.extend(['--cipher', cipher])
        key_size = kwargs.get('key_size', None)
        if key_size is not None:
            cmd.extend(['--key-size', key_size])
        cmd.extend([self.dev_name, self.dev_path])
        self._execute(*cmd, process_input=passphrase, check_exit_code=True, run_as_root=True, root_helper=self._root_helper)

    def attach_volume(self, context, **kwargs):
        """Shadow the device and pass an unencrypted version to the instance.

        Transparent disk encryption is achieved by mounting the volume via
        dm-crypt and passing the resulting device to the instance. The
        instance is unaware of the underlying encryption due to modifying the
        original symbolic link to refer to the device mounted by dm-crypt.
        """
        versionutils.report_deprecated_feature(LOG, "The plain CryptsetupEncryptor is deprecated and will be removed in a future release. Existing users are encouraged to retype any existing volumes using this encryptor to the 'luks' LuksEncryptor or 'luks2' Luks2Encryptor encryptors as soon as possible.")
        key = self._get_key(context).get_encoded()
        passphrase = self._get_passphrase(key)
        self._open_volume(passphrase, **kwargs)
        self._execute('ln', '--symbolic', '--force', '/dev/mapper/%s' % self.dev_name, self.symlink_path, root_helper=self._root_helper, run_as_root=True, check_exit_code=True)

    def _close_volume(self, **kwargs):
        """Closes the device (effectively removes the dm-crypt mapping)."""
        LOG.debug('closing encrypted volume %s', self.dev_path)
        self._execute('cryptsetup', 'remove', self.dev_name, run_as_root=True, check_exit_code=[0, 4], root_helper=self._root_helper)

    def detach_volume(self, **kwargs):
        """Removes the dm-crypt mapping for the device."""
        self._close_volume(**kwargs)

    def extend_volume(self, context, **kwargs):
        raise NotImplementedError()