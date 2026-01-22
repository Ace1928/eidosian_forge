from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.oneandone import (
def _remove_member(module, oneandone_conn, name, member_id):
    try:
        private_network_id = get_private_network(oneandone_conn, name)
        if module.check_mode:
            if private_network_id:
                network_member = oneandone_conn.get_private_network_server(private_network_id=private_network_id, server_id=member_id)
                if network_member:
                    return True
            return False
        network = oneandone_conn.remove_private_network_server(private_network_id=name, server_id=member_id)
        return network
    except Exception as ex:
        module.fail_json(msg=str(ex))