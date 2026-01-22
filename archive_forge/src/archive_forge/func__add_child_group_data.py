from __future__ import (absolute_import, division, print_function)
from ansible.plugins.inventory import BaseInventoryPlugin
from ansible_collections.dellemc.openmanage.plugins.module_utils.ome import RestOME
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import get_all_data_with_pagination
def _add_child_group_data(self, group_name, gdata):
    for child_name in gdata:
        self.inventory.add_child(group_name, child_name['Name'])