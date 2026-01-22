from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _enforce_count(self, module, clc):
    """
        Enforce that there is the right number of servers in the provided group.
        Starts or stops servers as necessary.
        :param module: the AnsibleModule object
        :param clc: the clc-sdk instance to use
        :return: a list of dictionaries with server information about the servers that were created or deleted
        """
    p = module.params
    changed = False
    count_group = p.get('count_group')
    datacenter = ClcServer._find_datacenter(clc, module)
    exact_count = p.get('exact_count')
    server_dict_array = []
    partial_servers_ids = []
    changed_server_ids = []
    if exact_count and count_group is None:
        return module.fail_json(msg="you must use the 'count_group' option with exact_count")
    servers, running_servers = ClcServer._find_running_servers_by_group(module, datacenter, count_group)
    if len(running_servers) == exact_count:
        changed = False
    elif len(running_servers) < exact_count:
        to_create = exact_count - len(running_servers)
        server_dict_array, changed_server_ids, partial_servers_ids, changed = self._create_servers(module, clc, override_count=to_create)
        for server in server_dict_array:
            running_servers.append(server)
    elif len(running_servers) > exact_count:
        to_remove = len(running_servers) - exact_count
        all_server_ids = sorted([x.id for x in running_servers])
        remove_ids = all_server_ids[0:to_remove]
        changed, server_dict_array, changed_server_ids = ClcServer._delete_servers(module, clc, remove_ids)
    return (server_dict_array, changed_server_ids, partial_servers_ids, changed)