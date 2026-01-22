from __future__ import (absolute_import, division, print_function)
from collections import defaultdict
import platform
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts import timeout
def _transform_name(self, key_name):
    if self.namespace:
        return self.namespace.transform(key_name)
    return key_name