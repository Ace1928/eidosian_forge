from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _object_service(connection, module):
    object_type = module.params['object_type']
    objects_service = _objects_service(connection, object_type)
    if object_type == 'system':
        return objects_service
    object_id = module.params['object_id']
    if object_id is None:
        sdk_object = search_by_name(objects_service, module.params['object_name'])
        if sdk_object is None:
            raise Exception("'%s' object '%s' was not found." % (module.params['object_type'], module.params['object_name']))
        object_id = sdk_object.id
    object_service = objects_service.service(object_id)
    if module.params['quota_name'] and object_type == 'data_center':
        quotas_service = object_service.quotas_service()
        return quotas_service.quota_service(get_id_by_name(quotas_service, module.params['quota_name']))
    return object_service