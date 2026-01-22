from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, PyVmomi, find_datacenter_by_name, get_all_objs
def __normalize_group_data(self, group_obj):
    """
        Return human readable group spec
        Args:
            group_obj: Group object

        Returns: Dictionary with DRS groups

        """
    if not all([group_obj]):
        return {}
    if hasattr(group_obj, 'host'):
        return dict(group_name=group_obj.name, hosts=self.__get_all_from_group(group_obj=group_obj, host_group=True), type='host')
    else:
        return dict(group_name=group_obj.name, vms=self.__get_all_from_group(group_obj=group_obj), type='vm')