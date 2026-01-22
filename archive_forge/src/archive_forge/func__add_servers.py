from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def _add_servers(module, oneandone_conn, name, members):
    try:
        private_network_id = get_private_network(oneandone_conn, name)
        if module.check_mode:
            if private_network_id and members:
                return True
            return False
        network = oneandone_conn.attach_private_network_servers(private_network_id=private_network_id, server_ids=members)
        return network
    except Exception as e:
        module.fail_json(msg=str(e))