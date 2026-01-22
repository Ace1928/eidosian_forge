from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def _delete_monitoring_policy_process(module, oneandone_conn, monitoring_policy_id, process_id):
    """
    Removes a process from a monitoring policy.
    """
    try:
        if module.check_mode:
            process = oneandone_conn.get_monitoring_policy_process(monitoring_policy_id=monitoring_policy_id, process_id=process_id)
            if process:
                return True
            return False
        monitoring_policy = oneandone_conn.delete_monitoring_policy_process(monitoring_policy_id=monitoring_policy_id, process_id=process_id)
        return monitoring_policy
    except Exception as ex:
        module.fail_json(msg=str(ex))