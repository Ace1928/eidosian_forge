from __future__ import absolute_import, division, print_function
import os.path
import platform
import shlex
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def unload_module_permanent(self):
    if self.module_is_loaded_persistently:
        self.disable_module_permanent()
        self.changed = True
    if self.permanent_params:
        self.disable_old_params()
        self.changed = True