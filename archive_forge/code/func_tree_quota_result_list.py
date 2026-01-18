from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def tree_quota_result_list(entity):
    """ Get the id and path associated with the Unity quota trees """
    result = []
    if entity:
        LOG.info(SUCCESSFULL_LISTED_MSG)
        for item in entity:
            result.append({'path': item.path, 'id': item.id})
        return result
    else:
        return None