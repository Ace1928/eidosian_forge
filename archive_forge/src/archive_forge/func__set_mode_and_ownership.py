from __future__ import absolute_import, division, print_function
import os
import time
from datetime import datetime
from ansible.module_utils.urls import urlparse
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def _set_mode_and_ownership(self):
    url = 'https://{0}:{1}/mgmt/tm/util/bash'.format(self.client.provider['server'], self.client.provider['server_port'])
    ownership = 'root:root'
    image_path = '/shared/images/{0}'.format(self.want.filename)
    file_mode = '0644'
    args = dict(command='run', utilCmdArgs='-c "chown {0} {1};chmod {2} {1}"'.format(ownership, image_path, file_mode))
    self.client.api.post(url, json=args)