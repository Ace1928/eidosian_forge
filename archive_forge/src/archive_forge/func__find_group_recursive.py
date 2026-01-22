from __future__ import absolute_import, division, print_function
import json
import os
import time
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
@staticmethod
def _find_group_recursive(module, group_list, lookup_group):
    """
        Find a server group by recursively walking the tree
        :param module: the AnsibleModule instance to use
        :param group_list: a list of groups to search
        :param lookup_group: the group to look for
        :return: list of groups
        """
    result = None
    for group in group_list.groups:
        subgroups = group.Subgroups()
        try:
            return subgroups.Get(lookup_group)
        except CLCException:
            result = ClcServer._find_group_recursive(module, subgroups, lookup_group)
        if result is not None:
            break
    return result