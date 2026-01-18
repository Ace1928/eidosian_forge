from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.static_routes.static_routes import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.static_routes import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.utils.utils import (
def update_netmask_to_cidr(address, netmask):
    dest = address + '/' + netmask_to_cidr(netmask)
    return dest