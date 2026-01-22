from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def _handle_allocation_state(self, pool, state=None):
    allocation_state = state or self.module.params.get('allocation_state')
    if not allocation_state:
        return pool
    if self.allocation_states.get(pool['state']) == allocation_state:
        return pool
    elif allocation_state in ['enabled', 'disabled']:
        pool = self._cancel_maintenance(pool)
        pool = self._update_storage_pool(pool=pool, allocation_state=allocation_state)
    elif allocation_state == 'maintenance':
        pool = self._update_storage_pool(pool=pool, allocation_state='enabled')
        pool = self._enable_maintenance(pool=pool)
    return pool