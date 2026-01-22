import binascii
import os
from oslo_concurrency import processutils
from oslo_log import log as logging
from os_brick.encryptors import base
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
def _format_luks_volume(self, passphrase, version, **kwargs):
    """Creates a LUKS header of a given version or type on the volume.

        :param passphrase: the passphrase used to access the volume
        :param version: the LUKS version or type to use: one of `luks`,
                        `luks1`, or `luks2`.  Be aware that `luks` gives you
                        the default LUKS format preferred by the particular
                        cryptsetup being used (depends on version and compile
                        time parameters), which could be either LUKS1 or
                        LUKS2, so it's better to be specific about what you
                        want here
        """
    LOG.debug('formatting encrypted volume %s', self.dev_path)
    cmd = ['cryptsetup', '--batch-mode', 'luksFormat', '--type', version, '--key-file=-']
    cipher = kwargs.get('cipher', None)
    if cipher is not None:
        cmd.extend(['--cipher', cipher])
    key_size = kwargs.get('key_size', None)
    if key_size is not None:
        cmd.extend(['--key-size', key_size])
    cmd.extend([self.dev_path])
    self._execute(*cmd, process_input=passphrase, check_exit_code=True, run_as_root=True, root_helper=self._root_helper, attempts=3)