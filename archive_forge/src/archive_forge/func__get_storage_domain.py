from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_storage_domain(self):
    """
        Gets the storage domain

        :return: otypes.StorageDomain or None
        """
    storage_domain_name = self._module.params.get('storage_domain')
    storage_domains_service = self._connection.system_service().storage_domains_service()
    return get_entity(storage_domains_service.storage_domain_service(get_id_by_name(storage_domains_service, storage_domain_name)))