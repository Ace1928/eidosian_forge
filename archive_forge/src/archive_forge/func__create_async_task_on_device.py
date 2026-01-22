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
def _create_async_task_on_device(self):
    """Creates an async cli script task in the REST API

        Returns:
            int: The ID of the task staged for running.

        :return:
        """
    command = ' '.join(self.want.api_params().values())
    args = {'command': 'run', 'name': '__ansible_mkqkview', 'utilCmdArgs': '/usr/bin/qkview {0}'.format(command)}
    uri = 'https://{0}:{1}/mgmt/tm/task/cli/script'.format(self.client.provider['server'], self.client.provider['server_port'])
    resp = self.client.api.post(uri, json=args)
    try:
        response = resp.json()
        return response['_taskId']
    except ValueError:
        raise F5ModuleError('Failed to create the async task on the device.')