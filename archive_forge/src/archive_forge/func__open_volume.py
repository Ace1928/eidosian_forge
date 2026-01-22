import binascii
import os
from oslo_concurrency import processutils
from oslo_log import log as logging
from os_brick.encryptors import base
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
def _open_volume(self, passphrase, **kwargs):
    """Opens the LUKS partition on the volume using passphrase.

        :param passphrase: the passphrase used to access the volume
        """
    LOG.debug('opening encrypted volume %s', self.dev_path)
    self._execute('cryptsetup', 'luksOpen', '--key-file=-', self.dev_path, self.dev_name, process_input=passphrase, run_as_root=True, check_exit_code=True, root_helper=self._root_helper)