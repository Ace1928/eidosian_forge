from __future__ import absolute_import, division, print_function
import re
import time
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _device_group_exists(self):
    errors = [401, 403, 409, 500, 501, 502, 503, 504]
    uri = 'https://{0}:{1}/mgmt/tm/cm/device-group/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], self.want.device_group)
    resp = self.client.api.get(uri)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status == 404 or ('code' in response and response['code'] == 404):
        return False
    if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
        return True
    if resp.status in errors or ('code' in response and response['code'] in errors):
        if 'message' in response:
            raise F5ModuleError(response['message'])
        else:
            raise F5ModuleError(resp.content)