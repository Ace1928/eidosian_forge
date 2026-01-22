import os
import re
import tempfile
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils.secretutils import md5
from os_brick import exception
from os_brick import executor
from os_brick.i18n import _
def _do_mount(self, mount_type, vz_share, mount_path, mount_options=None, flags=None):
    m = re.search('(?:(\\S+):\\/)?([a-zA-Z0-9_-]+)(?::(\\S+))?', vz_share)
    if not m:
        msg = _('Invalid Virtuozzo Storage share specification: %r.Must be: [MDS1[,MDS2],...:/]<CLUSTER NAME>[:PASSWORD].') % vz_share
        raise exception.BrickException(msg)
    mdss = m.group(1)
    cluster_name = m.group(2)
    passwd = m.group(3)
    if mdss:
        mdss = mdss.split(',')
        self._vzstorage_write_mds_list(cluster_name, mdss)
    if passwd:
        self._execute('pstorage', '-c', cluster_name, 'auth-node', '-P', process_input=passwd, root_helper=self._root_helper, run_as_root=True)
    mnt_cmd = ['pstorage-mount', '-c', cluster_name]
    if flags:
        mnt_cmd.extend(flags)
    mnt_cmd.extend([mount_path])
    self._execute(*mnt_cmd, root_helper=self._root_helper, run_as_root=True, check_exit_code=0)