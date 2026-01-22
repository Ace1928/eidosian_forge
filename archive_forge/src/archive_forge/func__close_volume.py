import binascii
import os
from oslo_concurrency import processutils
from oslo_log import log as logging
from os_brick.encryptors import base
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick import utils
def _close_volume(self, **kwargs):
    """Closes the device (effectively removes the dm-crypt mapping)."""
    LOG.debug('closing encrypted volume %s', self.dev_path)
    self._execute('cryptsetup', 'luksClose', self.dev_name, run_as_root=True, check_exit_code=[0, 4], root_helper=self._root_helper, attempts=3)