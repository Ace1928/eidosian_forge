from __future__ import (absolute_import, division, print_function)
import os
from collections import defaultdict
from json import loads
from ansible.errors import AnsibleError
from ansible.module_utils.urls import open_url
from ansible.inventory.group import to_safe_group_name
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
def _get_server_list(self):
    response = open_url(self.api_url + '/servers', headers={'Authorization': 'Bearer %s' % self.api_token})
    return loads(response.read())