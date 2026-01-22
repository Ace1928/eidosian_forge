from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _change_server_power_state(module, server, state):
    """
        Change the server powerState
        :param module: the module to check for intended state
        :param server: the server to start or stop
        :param state: the intended powerState for the server
        :return: the request object from clc-sdk call
        """
    result = None
    try:
        if state == 'started':
            result = server.PowerOn()
        else:
            result = server.ShutDown()
            if result and hasattr(result, 'requests') and result.requests[0]:
                return result
            else:
                result = server.PowerOff()
    except CLCException:
        module.fail_json(msg='Unable to change power state for server {0}'.format(server.id))
    return result