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
def _lvm_lv_remove(self, lv_name):
    """Remove an LV.

        :param lv_name: The name of the logical volume
        :type lv_name: ``str``
        """
    vg = self._get_lxc_vg()
    build_command = [self.module.get_bin_path('lvremove', True), '-f', '%s/%s' % (vg, lv_name)]
    rc, stdout, err = self.module.run_command(build_command)
    if rc != 0:
        self.failure(err=err, rc=rc, msg='Failed to remove LVM LV %s/%s' % (vg, lv_name), command=' '.join(build_command))