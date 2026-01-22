from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.openssh.backends.common import (
from ansible_collections.community.crypto.plugins.module_utils.openssh.certificate import (
def _is_partially_valid(self):
    return all([set(self.original_data.principals) == set(self.principals), self.original_data.signature_type == self.signature_algorithm if self.signature_algorithm else True, self.original_data.serial == self.serial_number if self.serial_number is not None else True, self.original_data.type == self.type, self._compare_time_parameters()])