from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.argspec.stp.stp import StpArgs
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def update_mstp(self, data):
    mstp_dict = {}
    mstp = data.get('mstp', None)
    if mstp:
        config = mstp.get('config', None)
        mst_instances = mstp.get('mst-instances', None)
        interfaces = mstp.get('interfaces', None)
        if config:
            mst_name = config.get('name', None)
            revision = config.get('revision', None)
            max_hop = config.get('max-hop', None)
            hello_time = config.get('hello-time', None)
            max_age = config.get('max-age', None)
            fwd_delay = config.get('forwarding-delay', None)
            if mst_name:
                mstp_dict['mst_name'] = mst_name
            if revision:
                mstp_dict['revision'] = revision
            if max_hop:
                mstp_dict['max_hop'] = max_hop
            if hello_time:
                mstp_dict['hello_time'] = hello_time
            if max_age:
                mstp_dict['max_age'] = max_age
            if fwd_delay:
                mstp_dict['fwd_delay'] = fwd_delay
        if mst_instances:
            mst_instance = mst_instances.get('mst-instance', None)
            if mst_instance:
                mst_instances_list = []
                for inst in mst_instance:
                    inst_dict = {}
                    mst_id = inst.get('mst-id', None)
                    config = inst.get('config', None)
                    interfaces = inst.get('interfaces', None)
                    if mst_id:
                        inst_dict['mst_id'] = mst_id
                    if interfaces:
                        intf_list = self.get_interfaces_list(interfaces)
                        if intf_list:
                            inst_dict['interfaces'] = intf_list
                    if config:
                        vlans = config.get('vlan', None)
                        bridge_priority = config.get('bridge-priority', None)
                        if vlans:
                            inst_dict['vlans'] = self.convert_vlans_list(vlans)
                        if bridge_priority:
                            inst_dict['bridge_priority'] = bridge_priority
                    if inst_dict:
                        mst_instances_list.append(inst_dict)
                if mst_instances_list:
                    mstp_dict['mst_instances'] = mst_instances_list
    return mstp_dict