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
@property
def virtual(self):
    result = dict()
    result['ltm:virtual::b487671f29ba'] = [dict(parameters=dict(name='virtual', destinationAddress=self.inbound_virtual['address'], mask=self.inbound_virtual['netmask'], destinationPort=self.inbound_virtual.get('port', 80)), subcollectionResources=self.profiles)]
    return result