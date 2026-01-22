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
def _lvm_lv_mount(self, lv_name, mount_point):
    """mount an lv.

        :param lv_name: name of the logical volume to mount
        :type lv_name: ``str``
        :param mount_point: path on the file system that is mounted.
        :type mount_point: ``str``
        """
    vg = self._get_lxc_vg()
    build_command = [self.module.get_bin_path('mount', True), '/dev/%s/%s' % (vg, lv_name), mount_point]
    rc, stdout, err = self.module.run_command(build_command)
    if rc != 0:
        self.failure(err=err, rc=rc, msg='failed to mountlvm lv %s/%s to %s' % (vg, lv_name, mount_point))