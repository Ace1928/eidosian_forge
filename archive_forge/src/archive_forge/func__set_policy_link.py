from __future__ import absolute_import, division, print_function
import os
import time
import tempfile
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _set_policy_link(self):
    policy_link = None
    uri = 'https://{0}:{1}/mgmt/tm/asm/policies/'.format(self.client.provider['server'], self.client.provider['server_port'])
    query = "?$filter=contains(name,'{0}')+and+contains(partition,'{1}')&$select=name,partition".format(self.want.name, self.want.partition)
    resp = self.client.api.get(uri + query)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if 'items' in response and response['items'] != []:
        for policy in response['items']:
            if policy['name'] == self.want.name and policy['partition'] == self.want.partition:
                policy_link = policy['selfLink']
    if not policy_link:
        raise F5ModuleError('The policy was not found')
    self.changes.update(dict(policyReference={'link': policy_link}))
    return True