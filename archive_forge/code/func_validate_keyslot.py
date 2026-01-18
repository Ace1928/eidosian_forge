from __future__ import (absolute_import, division, print_function)
import os
import re
import stat
from ansible.module_utils.basic import AnsibleModule
def validate_keyslot(self, param, luks_type):
    if self._module.params[param] is not None:
        if luks_type is None and param == 'keyslot':
            if 8 <= self._module.params[param] <= 31:
                self._module.fail_json(msg='You must specify type=luks2 when creating a new LUKS device to use keyslots 8-31.')
            elif not 0 <= self._module.params[param] <= 7:
                self._module.fail_json(msg='When not specifying a type, only the keyslots 0-7 are allowed.')
        if luks_type == 'luks1' and (not 0 <= self._module.params[param] <= 7):
            self._module.fail_json(msg='%s must be between 0 and 7 when using LUKS1.' % self._module.params[param])
        elif luks_type == 'luks2' and (not 0 <= self._module.params[param] <= 31):
            self._module.fail_json(msg='%s must be between 0 and 31 when using LUKS2.' % self._module.params[param])