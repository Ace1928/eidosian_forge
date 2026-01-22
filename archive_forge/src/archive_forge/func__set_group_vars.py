from __future__ import (absolute_import, division, print_function)
from ansible.plugins.inventory import BaseInventoryPlugin
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_all_data_with_pagination
def _set_group_vars(self, group):
    self.inventory.add_group(group)
    if 'group_vars' in self.config:
        group_vars = self.get_option('group_vars')
        if group in dict(group_vars):
            for key, val in dict(dict(group_vars)[group]).items():
                self.inventory.set_variable(group, key, val)