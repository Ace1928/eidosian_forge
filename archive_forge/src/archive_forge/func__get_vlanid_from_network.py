from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, TaskError, vmware_argument_spec, wait_for_task
from ansible_collections.community.vmware.plugins.module_utils.vm_device_helper import PyVmomiDeviceHelper
def _get_vlanid_from_network(self, network):
    """
        get the vlan id from network object
        :param network: network object to expect, either vim.Network or vim.dvs.DistributedVirtualPortgroup
        :return: vlan id as an integer
        :rtype: integer
        """
    vlan_id = None
    if isinstance(network, vim.dvs.DistributedVirtualPortgroup):
        vlan_id = network.config.defaultPortConfig.vlan.vlanId
    if isinstance(network, vim.Network) and hasattr(network, 'host'):
        for host in network.host:
            for pg in host.config.network.portgroup:
                if pg.spec.name == network.name:
                    vlan_id = pg.spec.vlanId
                    return vlan_id
    return vlan_id