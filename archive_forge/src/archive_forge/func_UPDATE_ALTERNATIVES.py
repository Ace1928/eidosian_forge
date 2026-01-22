from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
@property
def UPDATE_ALTERNATIVES(self):
    if self._UPDATE_ALTERNATIVES is None:
        self._UPDATE_ALTERNATIVES = self.module.get_bin_path('update-alternatives', True)
    return self._UPDATE_ALTERNATIVES