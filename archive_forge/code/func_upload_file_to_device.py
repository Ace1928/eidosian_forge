from __future__ import absolute_import, division, print_function
import hashlib
import os
import re
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import (
from ..module_utils.teem import send_teem
def upload_file_to_device(self, content, name):
    url = 'https://{0}:{1}/mgmt/shared/file-transfer/uploads'.format(self.client.provider['server'], self.client.provider['server_port'])
    try:
        upload_file(self.client, url, content, name)
    except F5ModuleError:
        raise F5ModuleError('Failed to upload the file.')