from __future__ import absolute_import, division, print_function
import re
import time
from datetime import datetime
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def update_node_offline_on_device(self):
    params = dict(session='user-disabled', state='user-down')
    uri = 'https://{0}:{1}/mgmt/tm/ltm/node/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name(self.want.partition, self.want.name))
    resp = self.client.api.patch(uri, json=params)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
        return True
    raise F5ModuleError(resp.content)