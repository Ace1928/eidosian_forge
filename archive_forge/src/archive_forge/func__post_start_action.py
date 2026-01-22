from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _post_start_action(self, entity):
    vm_service = self._service.service(entity.id)
    self._wait_for_UP(vm_service)
    self._attach_cd(vm_service.get())