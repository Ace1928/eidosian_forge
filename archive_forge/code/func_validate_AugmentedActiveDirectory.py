from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
def validate_AugmentedActiveDirectory(config):
    error = validate_ActiveDirectory(config, label='AugmentedActiveDirectory')
    if not error:
        return error
    for field in ('groupUIDAttribute', 'groupNameAttributes'):
        value = config.get(field)
        if not value:
            return 'AugmentedActiveDirectory: {0} is required'.format(field)
    groups_qry = config.get('groupsQuery')
    if not groups_qry or not isinstance(groups_qry, dict):
        return 'AugmentedActiveDirectory: groupsQuery requires as dictionnary.'
    isGroupDNOnly = config.get('groupUIDAttribute').strip() == 'dn'
    return validate_ldap_query(groups_qry, isDNOnly=isGroupDNOnly)