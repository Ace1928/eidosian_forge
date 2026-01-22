from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, TaskError, vmware_argument_spec, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def _get_network_object(self, vm_obj):
    """
        return network object matching given parameters
        :param vm_obj: vm object
        :return: network object
        :rtype: object
        """
    if not self.params['esxi_hostname'] or not self.params['cluster']:
        compute_resource = vm_obj.runtime.host
    else:
        compute_resource = self._get_compute_resource_by_name()
    pg_lookup = {}
    vlan_id = self.params['vlan_id']
    network_name = self.params['network_name']
    switch_name = self.params['switch']
    for pg in vm_obj.runtime.host.config.network.portgroup:
        pg_lookup[pg.spec.name] = {'switch': pg.spec.vswitchName, 'vlan_id': pg.spec.vlanId}
    if compute_resource:
        for network in compute_resource.network:
            if isinstance(network, vim.dvs.DistributedVirtualPortgroup):
                dvs = network.config.distributedVirtualSwitch
                if switch_name and dvs.config.name == switch_name or not switch_name:
                    if network.config.name == network_name:
                        return network
                    if hasattr(network.config.defaultPortConfig.vlan, 'vlanId') and network.config.defaultPortConfig.vlan.vlanId == vlan_id:
                        return network
                    if hasattr(network.config.defaultPortConfig.vlan, 'pvlanId') and network.config.defaultPortConfig.vlan.pvlanId == vlan_id:
                        return network
            elif isinstance(network, vim.Network):
                if network_name and network_name == network.name:
                    return network
                if vlan_id:
                    for k in pg_lookup.keys():
                        if vlan_id == pg_lookup[k]['vlan_id']:
                            if k == network.name:
                                return network
                            break
    return None