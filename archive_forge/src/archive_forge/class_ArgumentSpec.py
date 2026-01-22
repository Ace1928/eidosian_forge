from __future__ import absolute_import, division, print_function
import time
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import bigiq_version
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
class ArgumentSpec(object):

    def __init__(self):
        self.supports_check_mode = True
        argument_spec = dict(name=dict(required=True), description=dict(), servers=dict(type='list', elements='dict', options=dict(address=dict(required=True), port=dict(type='int', default=80))), inbound_virtual=dict(type='dict', options=dict(address=dict(required=True), netmask=dict(required=True), port=dict(type='int', default=80))), service_environment=dict(), add_analytics=dict(type='bool', default='no'), state=dict(default='present', choices=['present', 'absent']), wait=dict(type='bool', default='yes'))
        self.argument_spec = {}
        self.argument_spec.update(f5_argument_spec)
        self.argument_spec.update(argument_spec)