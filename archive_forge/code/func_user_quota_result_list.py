from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.unity.plugins.module_utils.storage.dell \
def user_quota_result_list(entity):
    """ Get the id and uid associated with the Unity user quotas """
    result = []
    if entity:
        LOG.info(SUCCESSFULL_LISTED_MSG)
        for item in entity:
            result.append({'uid': item.uid, 'id': item.id})
        return result
    else:
        return None