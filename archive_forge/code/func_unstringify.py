from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
def unstringify(val):
    if val == '-' or val == '':
        return None
    elif val == 'true':
        return True
    elif val == 'false':
        return False
    else:
        return val