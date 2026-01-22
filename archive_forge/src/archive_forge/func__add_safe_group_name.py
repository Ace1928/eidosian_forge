from __future__ import (absolute_import, division, print_function)
import socket
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_text
from ansible.plugins.inventory import BaseInventoryPlugin, Cacheable, to_safe_group_name
from ansible.module_utils.six import text_type
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _add_safe_group_name(self, group, child=None):
    group_name = self.inventory.add_group(to_safe_group_name('%s%s' % (self.get_option('group_prefix'), group.lower().replace(' ', ''))))
    if child is not None:
        self.inventory.add_child(group_name, child)
    return group_name