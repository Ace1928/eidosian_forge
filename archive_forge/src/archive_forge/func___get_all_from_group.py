from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi, find_datacenter_by_name, get_all_objs
def __get_all_from_group(self, group_obj, host_group=False):
    """
        Return all VM / Host names using given group
        Args:
            group_obj: Group object
            host_group: True if we want only host name from group

        Returns: List of VM / Host names belonging to given group object

        """
    obj_name_list = []
    if not all([group_obj]):
        return obj_name_list
    if not host_group and isinstance(group_obj, vim.cluster.VmGroup):
        obj_name_list = [vm.name for vm in group_obj.vm]
    elif host_group and isinstance(group_obj, vim.cluster.HostGroup):
        obj_name_list = [host.name for host in group_obj.host]
    return obj_name_list