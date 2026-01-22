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
def _delete_qkview(self):
    tpath_name = '{0}/{1}'.format(self.remote_dir, self.want.filename)
    params = dict(command='run', utilCmdArgs=tpath_name)
    uri = 'https://{0}:{1}/mgmt/tm/util/unix-rm'.format(self.client.provider['server'], self.client.provider['server_port'])
    resp = self.client.api.post(uri, json=params)
    try:
        response = resp.json()
    except ValueError:
        return False
    if resp.status == 404 or ('code' in response and response['code'] == 404):
        return False