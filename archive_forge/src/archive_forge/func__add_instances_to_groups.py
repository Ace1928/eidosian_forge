from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _add_instances_to_groups(self):
    """Add instance names to their dynamic inventory groups."""
    for instance in self.instances:
        self.inventory.add_host(make_unsafe(instance.label), group=instance.group)