from __future__ import (absolute_import, division, print_function)
from collections.abc import Mapping, MutableMapping
from enum import Enum
from itertools import chain
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars
def _get_hosts(self):
    hosts = []
    seen = {}
    for kid in self.get_descendants(include_self=True, preserve_ordering=True):
        kid_hosts = kid.hosts
        for kk in kid_hosts:
            if kk not in seen:
                seen[kk] = 1
                if self.name == 'all' and kk.implicit:
                    continue
                hosts.append(kk)
    return hosts