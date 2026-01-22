from __future__ import (absolute_import, division, print_function)
import fnmatch
import os
import sys
import re
import itertools
import traceback
from operator import attrgetter
from random import shuffle
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleOptionsError, AnsibleParserError
from ansible.inventory.data import InventoryData
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.parsing.utils.addresses import parse_address
from ansible.plugins.loader import inventory_loader
from ansible.utils.helpers import deduplicate_list
from ansible.utils.path import unfrackpath
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars
from ansible.vars.plugins import get_vars_from_inventory_sources
def add_dynamic_group(self, host, result_item):
    """
        Helper function to add a group (if it does not exist), and to assign the
        specified host to that group.
        """
    changed = False
    if not result_item.get('refresh'):
        self._cached_dynamic_grouping.append((host, result_item))
    real_host = self.hosts.get(host.name)
    if real_host is None:
        if host.name == self.localhost.name:
            real_host = self.localhost
        elif not result_item.get('refresh'):
            raise AnsibleError('%s cannot be matched in inventory' % host.name)
        else:
            return
    group_name = result_item.get('add_group')
    parent_group_names = result_item.get('parent_groups', [])
    if group_name not in self.groups:
        group_name = self.add_group(group_name)
    for name in parent_group_names:
        if name not in self.groups:
            self.add_group(name)
            changed = True
    group = self._inventory.groups[group_name]
    for parent_group_name in parent_group_names:
        parent_group = self.groups[parent_group_name]
        new = parent_group.add_child_group(group)
        if new and (not changed):
            changed = True
    if real_host not in group.get_hosts():
        changed = group.add_host(real_host)
    if group not in real_host.get_groups():
        changed = real_host.add_group(group)
    if changed:
        self.reconcile_inventory()
    result_item['changed'] = changed