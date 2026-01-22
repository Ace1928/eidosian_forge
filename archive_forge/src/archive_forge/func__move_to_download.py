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
def _move_to_download(self):
    move_path = '/var/local/ucs/{0} {1}/{0}'.format(self.want.filename, self.remote_dir)
    params = dict(command='run', utilCmdArgs=move_path)
    uri = 'https://{0}:{1}/mgmt/tm/util/unix-mv/'.format(self.client.provider['server'], self.client.provider['server_port'])
    resp = self.client.api.post(uri, json=params)
    try:
        response = resp.json()
        if 'commandResult' in response:
            if 'cannot stat' in response['commandResult']:
                raise F5ModuleError(response['commandResult'])
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if 'code' in response and response['code'] in [400, 403]:
        if 'message' in response:
            raise F5ModuleError(response['message'])
        else:
            raise F5ModuleError(resp.content)
    return True