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
def _update_temporary_cli_script_on_device(self, args):
    uri = 'https://{0}:{1}/mgmt/tm/cli/script/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name('Common', '__ansible_mkqkview'))
    resp = self.client.api.put(uri, json=args)
    try:
        resp.json()
        return True
    except ValueError:
        raise F5ModuleError('Failed to update temporary cli script on device.')