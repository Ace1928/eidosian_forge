from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _action_save_configuration(self, entity):
    if not self._module.check_mode:
        self._service.service(entity.id).commit_net_config()
    self.changed = True