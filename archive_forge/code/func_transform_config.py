from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.lag_interfaces.lag_interfaces import Lag_interfacesArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible.module_utils.connection import ConnectionError
def transform_config(self, conf):
    trans_cfg = dict()
    trans_cfg['name'] = conf['name']
    trans_cfg['members'] = dict()
    if conf['ifname']:
        interfaces = list()
        interface = {'member': conf['ifname']}
        interfaces.append(interface)
        trans_cfg['members'] = {'interfaces': interfaces}
    return trans_cfg