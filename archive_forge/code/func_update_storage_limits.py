from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def update_storage_limits(self, entity):
    new_limits = {}
    for storage in self._module.params.get('storages'):
        new_limits[storage.get('name', '')] = {'size': storage.get('size')}
    old_limits = {}
    sd_limit_service = self._service.service(entity.id).quota_storage_limits_service()
    for limit in sd_limit_service.list():
        storage = get_link_name(self._connection, limit.storage_domain) if limit.storage_domain else ''
        old_limits[storage] = {'size': limit.limit}
        sd_limit_service.service(limit.id).remove()
    return new_limits == old_limits