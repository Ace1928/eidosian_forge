from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _filter_by_config(self):
    """Filter instances by user specified configuration."""
    regions = self.get_option('regions')
    if regions:
        self.instances = [instance for instance in self.instances if instance.region.id in regions]
    types = self.get_option('types')
    if types:
        self.instances = [instance for instance in self.instances if instance.type.id in types]
    tags = self.get_option('tags')
    if tags:
        self.instances = [instance for instance in self.instances if any((tag in instance.tags for tag in tags))]