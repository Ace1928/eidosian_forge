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
def _set_image_url(self, item):
    path = urlparse(item['selfLink']).path
    self.image_url = 'https://{0}:{1}{2}'.format(self.client.provider['server'], self.client.provider['server_port'], path)