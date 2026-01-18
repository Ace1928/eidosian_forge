from __future__ import absolute_import, division, print_function
import datetime
import fileinput
import os
import re
import shutil
import sys
from ansible.module_utils.basic import AnsibleModule
def update_sorcery(module):
    """ Update sorcery scripts.

    This runs 'sorcery update' ('sorcery -u'). Check mode always returns a
    positive change value.

    """
    changed = False
    if module.check_mode:
        return (True, 'would have updated Sorcery')
    else:
        sorcery_ver = get_sorcery_ver(module)
        cmd_sorcery = '%s update' % SORCERY['sorcery']
        rc, stdout, stderr = module.run_command(cmd_sorcery)
        if rc != 0:
            module.fail_json(msg='unable to update Sorcery: ' + stdout)
        if sorcery_ver != get_sorcery_ver(module):
            changed = True
        return (changed, 'successfully updated Sorcery')