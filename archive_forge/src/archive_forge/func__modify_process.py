from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def _modify_process(module, oneandone_conn, monitoring_policy_id, process_id, process):
    """
    Modifies a monitoring policy process.
    """
    try:
        if module.check_mode:
            cm_process = oneandone_conn.get_monitoring_policy_process(monitoring_policy_id=monitoring_policy_id, process_id=process_id)
            if cm_process:
                return True
            return False
        monitoring_policy_process = oneandone.client.Process(process=process['process'], alert_if=process['alert_if'], email_notification=process['email_notification'])
        monitoring_policy = oneandone_conn.modify_process(monitoring_policy_id=monitoring_policy_id, process_id=process_id, process=monitoring_policy_process)
        return monitoring_policy
    except Exception as ex:
        module.fail_json(msg=str(ex))