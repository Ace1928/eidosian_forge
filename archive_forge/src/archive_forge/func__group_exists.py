from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _group_exists(self, group_name, parent_name):
    """
        Check to see if a group exists
        :param group_name: string - the group to check
        :param parent_name: string - the parent of group_name
        :return: boolean - whether the group exists
        """
    result = False
    if group_name in self.group_dict:
        group, parent = self.group_dict[group_name]
        if parent_name is None or parent_name == parent.name:
            result = True
    return result