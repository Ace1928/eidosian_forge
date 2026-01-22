from __future__ import absolute_import, division, print_function
import copy
import datetime
import signal
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import exec_command
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.teem import send_teem
def _get_client_connection(self):
    return F5RestClient(**self.module.params)