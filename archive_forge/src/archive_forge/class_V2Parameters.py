from __future__ import absolute_import, division, print_function
import traceback
from datetime import datetime
from ansible.module_utils.basic import (
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
class V2Parameters(Parameters):
    updatables = ['snmp_version', 'community', 'destination', 'port', 'network']
    returnables = ['snmp_version', 'community', 'destination', 'port', 'network']
    api_attributes = ['version', 'community', 'host', 'port', 'network']

    @property
    def network(self):
        if self._values['network'] is None:
            return None
        network = str(self._values['network'])
        if network == 'management':
            return 'mgmt'
        elif network == 'default':
            return ''
        else:
            return network