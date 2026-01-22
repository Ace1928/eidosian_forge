from __future__ import absolute_import, division, print_function
import os
import time
import tempfile
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _move_binary_to_download(self):
    name = '{0}~{1}'.format(self.client.provider['user'], self.want.file)
    move_path = '/var/tmp/{0} {1}/{2}'.format(self.want.file, '/ts/var/rest', name)
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
    if resp.status in [200, 201] or ('code' in response and response['code'] in [200, 201]):
        return True
    raise F5ModuleError(resp.content)