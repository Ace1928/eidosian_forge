from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _refresh_servers(module, servers):
    """
        Loop through a list of servers and refresh them.
        :param module: the AnsibleModule object
        :param servers: list of clc-sdk.Server instances to refresh
        :return: none
        """
    for server in servers:
        try:
            server.Refresh()
        except CLCException as ex:
            module.fail_json(msg='Unable to refresh the server {0}. {1}'.format(server.id, ex.message))