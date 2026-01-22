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
def _get_vg_free_pe(self, vg_name):
    """Return the available size of a given VG.

        :param vg_name: Name of volume.
        :type vg_name: ``str``
        :returns: size and measurement of an LV
        :type: ``tuple``
        """
    build_command = ['vgdisplay', vg_name, '--units', 'g']
    rc, stdout, err = self.module.run_command(build_command)
    if rc != 0:
        self.failure(err=err, rc=rc, msg='failed to read vg %s' % vg_name, command=' '.join(build_command))
    vg_info = [i.strip() for i in stdout.splitlines()][1:]
    free_pe = [i for i in vg_info if i.startswith('Free')]
    _free_pe = free_pe[0].split()
    return (float(_free_pe[-2]), _free_pe[-1])