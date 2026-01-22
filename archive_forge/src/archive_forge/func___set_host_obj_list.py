from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def __set_host_obj_list(self, host_list=None):
    """
        Function populate host object list from list of hostnames
        Args:
            host_list: List of host names

        Returns: None

        """
    if host_list is None:
        host_list = self.__host_list
    if host_list is not None:
        for host in host_list:
            if not self.module.check_mode:
                host_obj = self.find_hostsystem_by_name(host)
                if host_obj is None:
                    raise Exception('ESXi host %s does not exist in cluster %s' % (host, self.__cluster_name))
                self.__host_obj_list.append(host_obj)