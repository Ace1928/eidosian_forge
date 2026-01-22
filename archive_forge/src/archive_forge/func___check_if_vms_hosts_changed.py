from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def __check_if_vms_hosts_changed(self, group_name=None, cluster_obj=None, host_group=False):
    """
        Function to check if VMs/Hosts changed
        Args:
            group_name: Name of group
            cluster_obj: vim Cluster object
            host_group: True if we want to check hosts, else check vms

        Returns: Bool

        """
    if group_name is None:
        group_name = self.__group_name
    if cluster_obj is None:
        cluster_obj = self.__cluster_obj
    list_a = self.__host_list if host_group else self.__vm_list
    list_b = self.__populate_vm_host_list(host_group=host_group)
    if set(list_a) == set(list_b):
        return False
    return True