from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.icontrol import tmos_version
from ..module_utils.teem import send_teem
def _get_validated_ip_address(self, address):
    if is_valid_ip(self._values[address]):
        return self._values[address]
    raise F5ModuleError("The specified '{0}' is not a valid IP address".format(address))