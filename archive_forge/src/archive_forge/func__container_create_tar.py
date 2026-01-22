from __future__ import absolute_import, division, print_function
import os
import os.path
import re
import shutil
import subprocess
import tempfile
import time
import shlex
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE
from ansible.module_utils.common.text.converters import to_text, to_bytes
def _container_create_tar(self):
    """Create a tar archive from an LXC container.

        The process is as follows:
            * Stop or Freeze the container
            * Create temporary dir
            * Copy container and config to temporary directory
            * If LVM backed:
                * Create LVM snapshot of LV backing the container
                * Mount the snapshot to tmpdir/rootfs
            * Restore the state of the container
            * Create tar of tmpdir
            * Clean up
        """
    temp_dir = tempfile.mkdtemp()
    work_dir = os.path.join(temp_dir, self.container_name)
    lxc_rootfs = self.container.get_config_item('lxc.rootfs')
    block_backed = lxc_rootfs.startswith(os.path.join(os.sep, 'dev'))
    overlayfs_backed = lxc_rootfs.startswith('overlayfs')
    mount_point = os.path.join(work_dir, 'rootfs')
    snapshot_name = '%s_lxc_snapshot' % self.container_name
    container_state = self._get_state()
    try:
        if container_state not in ['stopped', 'frozen']:
            if container_state == 'running':
                self.container.freeze()
            else:
                self.container.stop()
        self._rsync_data(lxc_rootfs, temp_dir)
        if block_backed:
            if snapshot_name not in self._lvm_lv_list():
                if not os.path.exists(mount_point):
                    os.makedirs(mount_point)
                size, measurement = self._get_lv_size(lv_name=self.container_name)
                self._lvm_snapshot_create(source_lv=self.container_name, snapshot_name=snapshot_name, snapshot_size_gb=size)
                self._lvm_lv_mount(lv_name=snapshot_name, mount_point=mount_point)
            else:
                self.failure(err='snapshot [ %s ] already exists' % snapshot_name, rc=1, msg='The snapshot [ %s ] already exists. Please clean up old snapshot of containers before continuing.' % snapshot_name)
        elif overlayfs_backed:
            lowerdir, upperdir = lxc_rootfs.split(':')[1:]
            self._overlayfs_mount(lowerdir=lowerdir, upperdir=upperdir, mount_point=mount_point)
        self.state_change = True
        return self._create_tar(source_dir=work_dir)
    finally:
        if block_backed or overlayfs_backed:
            self._unmount(mount_point)
        if block_backed:
            self._lvm_lv_remove(snapshot_name)
        if container_state == 'running':
            if self._get_state() == 'frozen':
                self.container.unfreeze()
            else:
                self.container.start()
        shutil.rmtree(temp_dir)