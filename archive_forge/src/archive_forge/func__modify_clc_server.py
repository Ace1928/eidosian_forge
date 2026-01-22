from __future__ import absolute_import, division, print_function
import json
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _modify_clc_server(clc, module, server_id, cpu, memory):
    """
        Modify the memory or CPU of a clc server.
        :param clc: the clc-sdk instance to use
        :param module: the AnsibleModule object
        :param server_id: id of the server to modify
        :param cpu: the new cpu value
        :param memory: the new memory value
        :return: the result of CLC API call
        """
    result = None
    acct_alias = clc.v2.Account.GetAlias()
    try:
        job_obj = clc.v2.API.Call('PATCH', 'servers/%s/%s' % (acct_alias, server_id), json.dumps([{'op': 'set', 'member': 'memory', 'value': memory}, {'op': 'set', 'member': 'cpu', 'value': cpu}]))
        result = clc.v2.Requests(job_obj)
    except APIFailedResponse as ex:
        module.fail_json(msg='Unable to update the server configuration for server : "{0}". {1}'.format(server_id, str(ex.response_text)))
    return result