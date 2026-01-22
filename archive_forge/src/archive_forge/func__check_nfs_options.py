import os
import re
import tempfile
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils.secretutils import md5
from os_brick import exception
from os_brick import executor
from os_brick.i18n import _
def _check_nfs_options(self):
    """Checks and prepares nfs mount type options."""
    self._nfs_mount_type_opts = {'nfs': self._mount_options}
    nfs_vers_opt_patterns = ['^nfsvers', '^vers', '^v[\\d]']
    for opt in nfs_vers_opt_patterns:
        if self._option_exists(self._mount_options, opt):
            return
    pnfs_opts = self._update_option(self._mount_options, 'vers', '4')
    pnfs_opts = self._update_option(pnfs_opts, 'minorversion', '1')
    self._nfs_mount_type_opts['pnfs'] = pnfs_opts