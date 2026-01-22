from __future__ import (absolute_import, division, print_function)
from collections.abc import Mapping
from ansible import constants as C
from ansible.template import Templar, AnsibleUndefined
def _find_host(self, host_name):
    return self._inventory.get_host(host_name)