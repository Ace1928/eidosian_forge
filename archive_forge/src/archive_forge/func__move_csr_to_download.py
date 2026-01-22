from __future__ import absolute_import, division, print_function
import os
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _move_csr_to_download(self):
    uri = 'https://{0}:{1}/mgmt/tm/util/unix-mv/'.format(self.client.provider['server'], self.client.provider['server_port'])
    args = dict(command='run', utilCmdArgs='/config/ssl/ssl.csr/{0} {1}/{0}'.format(self.want.name, self.remote_dir))
    self.client.api.post(uri, json=args)
    return True