from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def update_roles(role_facts, cursor, role, existing, required):
    for assigned_role in set(existing) - set(required):
        cursor.execute('revoke {0} from {1}'.format(assigned_role, role))
    for assigned_role in set(required) - set(existing):
        cursor.execute('grant {0} to {1}'.format(assigned_role, role))