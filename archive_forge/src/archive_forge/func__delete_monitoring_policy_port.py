from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def _delete_monitoring_policy_port(module, oneandone_conn, monitoring_policy_id, port_id):
    """
    Removes a port from a monitoring policy.
    """
    try:
        if module.check_mode:
            monitoring_policy = oneandone_conn.delete_monitoring_policy_port(monitoring_policy_id=monitoring_policy_id, port_id=port_id)
            if monitoring_policy:
                return True
            return False
        monitoring_policy = oneandone_conn.delete_monitoring_policy_port(monitoring_policy_id=monitoring_policy_id, port_id=port_id)
        return monitoring_policy
    except Exception as ex:
        module.fail_json(msg=str(ex))