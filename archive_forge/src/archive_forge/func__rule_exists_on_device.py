from __future__ import absolute_import, division, print_function
import re
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _rule_exists_on_device(self, rule_name, draft=False):
    if draft:
        uri = 'https://{0}:{1}/mgmt/tm/ltm/policy/{2}/rules/{3}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name, sub_path='Drafts'), quote_plus(rule_name))
    else:
        uri = 'https://{0}:{1}/mgmt/tm/ltm/policy/{2}/rules/{3}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name), self.want.name)
    resp = self.client.api.get(uri)
    try:
        response = resp.json()
    except ValueError:
        return False
    if resp.status == 404 or ('code' in response and response['code'] == 404):
        return False
    return True