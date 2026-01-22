from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _create_rule_on_device(self, rule_name, idx, draft=False):
    params = dict(name=rule_name, ordinal=idx)
    if draft:
        uri = 'https://{0}:{1}/mgmt/tm/ltm/policy/{2}/rules/'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name, sub_path='Drafts'))
    else:
        uri = 'https://{0}:{1}/mgmt/tm/ltm/policy/{2}/rules/'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
    resp = self.client.api.post(uri, json=params)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if 'code' in response and response['code'] in [400, 403, 409]:
        if 'message' in response:
            raise F5ModuleError(response['message'])
        else:
            raise F5ModuleError(resp.content)