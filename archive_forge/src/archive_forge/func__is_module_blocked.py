from __future__ import absolute_import, division, print_function
import os
import re
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
def _is_module_blocked(self):
    for line in self.vars.lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if self.pattern.match(stripped):
            return True
    return False