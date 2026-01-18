from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
import traceback
import datetime
def user_role_assignment_build_rfc_params(roles, username):
    rfc_table = []
    for role_name in roles:
        table_row = {'AGR_NAME': role_name}
        add_to_dict(table_row, 'FROM_DAT', datetime.date.today())
        add_to_dict(table_row, 'TO_DAT', '20991231')
        rfc_table.append(table_row)
    return {'USERNAME': username, 'ACTIVITYGROUPS': rfc_table}