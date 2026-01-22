from __future__ import (absolute_import, division, print_function)
import json
import ssl
from time import sleep
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def _add_vms(self, vms, hosts, pools):
    for uuid, vm in vms.items():
        group = 'with_ip'
        ip = vm.get('mainIpAddress')
        entry_name = uuid
        power_state = vm['power_state'].lower()
        pool_name = self._pool_group_name_for_uuid(pools, vm['$poolId'])
        host_name = self._host_group_name_for_uuid(hosts, vm['$container'])
        self.inventory.add_host(entry_name)
        self.inventory.add_child(power_state, entry_name)
        if host_name:
            self.inventory.add_child(host_name, entry_name)
        if pool_name:
            self.inventory.add_child(pool_name, entry_name)
        if ip is None:
            group = 'without_ip'
        self.inventory.add_group(group)
        self.inventory.add_child(group, entry_name)
        self.inventory.set_variable(entry_name, 'uuid', uuid)
        self.inventory.set_variable(entry_name, 'ip', ip)
        self.inventory.set_variable(entry_name, 'ansible_host', ip)
        self.inventory.set_variable(entry_name, 'power_state', power_state)
        self.inventory.set_variable(entry_name, 'name_label', vm['name_label'])
        self.inventory.set_variable(entry_name, 'type', vm['type'])
        self.inventory.set_variable(entry_name, 'cpus', vm['CPUs']['number'])
        self.inventory.set_variable(entry_name, 'tags', vm['tags'])
        self.inventory.set_variable(entry_name, 'memory', vm['memory']['size'])
        self.inventory.set_variable(entry_name, 'has_ip', group == 'with_ip')
        self.inventory.set_variable(entry_name, 'is_managed', vm.get('managementAgentDetected', False))
        self.inventory.set_variable(entry_name, 'os_version', vm['os_version'])
        self._apply_constructable(entry_name, self.inventory.get_host(entry_name).get_vars())