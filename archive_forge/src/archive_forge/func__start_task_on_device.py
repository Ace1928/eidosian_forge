from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _start_task_on_device(self, task):
    payload = {'_taskState': 'VALIDATING'}
    uri = 'https://{0}:{1}/mgmt/tm/task/sys/ucs/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], task)
    resp = self.client.api.put(uri, json=payload)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if resp.status in [200, 201, 202] or ('code' in response and response['code'] in [200, 201, 202]):
        return True
    raise F5ModuleError(resp.content)