from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
import traceback
import datetime
def user_profile_assignment_build_rfc_params(profiles, username):
    rfc_table = []
    for profile_name in profiles:
        table_row = {'BAPIPROF': profile_name}
        rfc_table.append(table_row)
    return {'USERNAME': username, 'PROFILES': rfc_table}