from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _get_group_tree_for_datacenter(self, datacenter=None):
    """
        Walk the tree of groups for a datacenter
        :param datacenter: string - the datacenter to walk (ex: 'UC1')
        :return: a dictionary of groups and parents
        """
    self.root_group = self.clc.v2.Datacenter(location=datacenter).RootGroup()
    return self._walk_groups_recursive(parent_group=None, child_group=self.root_group)