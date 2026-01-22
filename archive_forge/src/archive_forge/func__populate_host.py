from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import missing_required_lib
from ..module_utils.gcp_utils import (
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
def _populate_host(self, item):
    """
        :param item: A GCP instance
        """
    hostname = item.hostname()
    self.inventory.add_host(hostname)
    for key in item.to_json():
        try:
            self.inventory.set_variable(hostname, self.get_option('vars_prefix') + key, item.to_json()[key])
        except (ValueError, TypeError) as e:
            self.display.warning('Could not set host info hostvar for %s, skipping %s: %s' % (hostname, key, to_text(e)))
    self.inventory.add_child('all', hostname)