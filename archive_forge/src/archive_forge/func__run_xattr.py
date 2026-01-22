from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def _run_xattr(module, cmd, check_rc=True):
    try:
        rc, out, err = module.run_command(cmd, check_rc=check_rc)
    except Exception as e:
        module.fail_json(msg='%s!' % to_native(e))
    result = {}
    for line in out.splitlines():
        if line.startswith('#') or line == '':
            pass
        elif '=' in line:
            key, val = line.split('=', 1)
            result[key] = val.strip('"')
        else:
            result[line] = ''
    return result