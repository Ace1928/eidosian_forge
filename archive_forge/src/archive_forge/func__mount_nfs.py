import os
import re
import tempfile
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils.secretutils import md5
from os_brick import exception
from os_brick import executor
from os_brick.i18n import _
def _mount_nfs(self, nfs_share, mount_path, flags=None):
    """Mount nfs share using present mount types."""
    mnt_errors = {}
    for mnt_type in sorted(self._nfs_mount_type_opts.keys(), reverse=True):
        options = self._nfs_mount_type_opts[mnt_type]
        try:
            self._do_mount('nfs', nfs_share, mount_path, options, flags)
            LOG.debug('Mounted %(sh)s using %(mnt_type)s.', {'sh': nfs_share, 'mnt_type': mnt_type})
            return
        except Exception as e:
            mnt_errors[mnt_type] = str(e)
            LOG.debug('Failed to do %s mount.', mnt_type)
    raise exception.BrickException(_('NFS mount failed for share %(sh)s. Error - %(error)s') % {'sh': nfs_share, 'error': mnt_errors})