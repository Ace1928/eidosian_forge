from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_migratable(self):
    if self.param('migratable') is not None:
        return self.param('migratable')
    if self.param('pass_through') == 'enabled':
        return True