from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
def __get_group_by_name(self, group_name=None, cluster_obj=None):
    """
        Function to get group by name
        Args:
            group_name: Name of group
            cluster_obj: vim Cluster object

        Returns: Group Object if found or None

        """
    if group_name is None:
        group_name = self.__group_name
    if cluster_obj is None:
        cluster_obj = self.__cluster_obj
    if self.module.check_mode and cluster_obj is None:
        return None
    for group in cluster_obj.configurationEx.group:
        if group.name == group_name:
            return group
    return None