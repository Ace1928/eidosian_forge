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
def _remove_temporary_cli_script_from_device(self):
    uri = 'https://{0}:{1}/mgmt/tm/task/cli/script/{2}'.format(self.client.provider['server'], self.client.provider['server_port'], transform_name('Common', '__ansible_mkqkview'))
    try:
        self.client.api.delete(uri)
        return True
    except ValueError:
        raise F5ModuleError('Failed to remove the temporary cli script from the device.')