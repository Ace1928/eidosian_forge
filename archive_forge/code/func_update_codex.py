from __future__ import absolute_import, division, print_function
import datetime
import fileinput
import os
import re
import shutil
import sys
from ansible.module_utils.basic import AnsibleModule
def update_codex(module):
    """ Update grimoire collections.

    This runs 'scribe update'. Check mode always returns a positive change
    value when 'cache_valid_time' is used.

    """
    params = module.params
    changed = False
    codex = codex_list(module)
    fresh = codex_fresh(codex, module)
    if module.check_mode:
        if not fresh:
            changed = True
        return (changed, 'would have updated Codex')
    else:
        if not fresh:
            module.run_command_environ_update.update(dict(SILENT='1'))
            cmd_scribe = '%s update' % SORCERY['scribe']
            if params['repository']:
                cmd_scribe += ' %s' % ' '.join(codex.keys())
            rc, stdout, stderr = module.run_command(cmd_scribe)
            if rc != 0:
                module.fail_json(msg='unable to update Codex: ' + stdout)
            if codex != codex_list(module):
                changed = True
        return (changed, 'successfully updated Codex')