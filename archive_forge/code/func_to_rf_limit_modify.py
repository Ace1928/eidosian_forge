from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.dellemc.powerflex.plugins.module_utils.storage.dell \
def to_rf_limit_modify(pd_details, rf_cache_limits):
    """
    Check if modification required for RF cache for protection domain
    :param pd_details: Details of the protection domain
    :type pd_details: dict
    :param rf_cache_limits: dict for RF cache
    :type rf_cache_limits: dict
    :return: Dictionary containing the attributes of protection domain
             which are to be updated
    :rtype: dict
    """
    modify_dict = {}
    if rf_cache_limits is not None:
        modify_dict['is_enabled'] = None
        modify_dict['page_size'] = None
        modify_dict['max_io_limit'] = None
        modify_dict['pass_through_mode'] = None
        if rf_cache_limits['is_enabled'] is not None and pd_details['rfcacheEnabled'] != rf_cache_limits['is_enabled']:
            modify_dict['is_enabled'] = rf_cache_limits['is_enabled']
        if rf_cache_limits['page_size'] is not None and pd_details['rfcachePageSizeKb'] != rf_cache_limits['page_size']:
            modify_dict['page_size'] = rf_cache_limits['page_size']
        if rf_cache_limits['max_io_limit'] is not None and pd_details['rfcacheMaxIoSizeKb'] != rf_cache_limits['max_io_limit']:
            modify_dict['max_io_limit'] = rf_cache_limits['max_io_limit']
        if rf_cache_limits['pass_through_mode'] is not None and pd_details['rfcacheOpertionalMode'] != rf_cache_limits['pass_through_mode']:
            modify_dict['pass_through_mode'] = rf_cache_limits['pass_through_mode']
    return modify_dict