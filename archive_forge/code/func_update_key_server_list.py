from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def update_key_server_list(self, current):
    desired_servers = self.na_helper.safe_get(self.parameters, ['external', 'servers'])
    if desired_servers is None:
        return
    desired_servers = [server['server'] for server in desired_servers]
    current_servers = self.na_helper.safe_get(current, ['external', 'servers']) or []
    current_servers = [server['server'] for server in current_servers]
    for server in current_servers:
        if server not in desired_servers:
            self.remove_external_server_rest(server)
    for server in desired_servers:
        if server not in current_servers:
            self.add_external_server_rest(server)