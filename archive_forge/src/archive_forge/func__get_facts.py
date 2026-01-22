from __future__ import (absolute_import, division, print_function)
import copy
import json
from ansible_collections.theforeman.foreman.plugins.module_utils._version import LooseVersion
from time import sleep
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible.module_utils.common._collections_compat import MutableMapping
from ansible.plugins.inventory import BaseInventoryPlugin, Cacheable, to_safe_group_name, Constructable
def _get_facts(self, host):
    """Fetch all host facts of the host"""
    ret = self._get_facts_by_id(host['id'])
    if len(ret.values()) == 0:
        facts = {}
    elif len(ret.values()) == 1:
        facts = list(ret.values())[0]
    else:
        raise ValueError("More than one set of facts returned for '%s'" % host)
    return facts