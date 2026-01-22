from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _find_server_by_uuid_w_retry(clc, module, svr_uuid, alias=None, retries=5, back_out=2):
    """
        Find the clc server by the UUID returned from the provisioning request.  Retry the request if a 404 is returned.
        :param clc: the clc-sdk instance to use
        :param module: the AnsibleModule object
        :param svr_uuid: UUID of the server
        :param retries: the number of retry attempts to make prior to fail. default is 5
        :param alias: the Account Alias to search
        :return: a clc-sdk.Server instance
        """
    if not alias:
        alias = clc.v2.Account.GetAlias()
    while True:
        retries -= 1
        try:
            server_obj = clc.v2.API.Call(method='GET', url='servers/%s/%s?uuid=true' % (alias, svr_uuid))
            server_id = server_obj['id']
            server = clc.v2.Server(id=server_id, alias=alias, server_obj=server_obj)
            return server
        except APIFailedResponse as e:
            if e.response_status_code != 404:
                return module.fail_json(msg='A failure response was received from CLC API when attempting to get details for a server:  UUID=%s, Code=%i, Message=%s' % (svr_uuid, e.response_status_code, e.message))
            if retries == 0:
                return module.fail_json(msg='Unable to reach the CLC API after 5 attempts')
            time.sleep(back_out)
            back_out *= 2