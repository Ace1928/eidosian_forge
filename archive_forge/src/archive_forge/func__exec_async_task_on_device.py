from __future__ import absolute_import, division, print_function
import os
import re
import socket
import ssl
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _exec_async_task_on_device(self, task_id):
    args = {'_taskState': 'VALIDATING'}
    uri = 'https://{0}:{1}/mgmt/tm/task/cli/script/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], task_id)
    resp = self.client.api.put(uri, json=args)
    try:
        resp.json()
        return True
    except ValueError:
        raise F5ModuleError('Failed to execute the async task on the device')