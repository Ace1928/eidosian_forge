from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def _attach_monitoring_policy_server(module, oneandone_conn, monitoring_policy_id, servers):
    """
    Attaches servers to a monitoring policy.
    """
    try:
        attach_servers = []
        for _server_id in servers:
            server_id = get_server(oneandone_conn, _server_id)
            attach_server = oneandone.client.AttachServer(server_id=server_id)
            attach_servers.append(attach_server)
        if module.check_mode:
            if attach_servers:
                return True
            return False
        monitoring_policy = oneandone_conn.attach_monitoring_policy_server(monitoring_policy_id=monitoring_policy_id, servers=attach_servers)
        return monitoring_policy
    except Exception as ex:
        module.fail_json(msg=str(ex))