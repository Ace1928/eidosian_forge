from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_port_mirroring(self):
    if self.param('pass_through') == 'enabled':
        return False
    return self.param('port_mirroring')