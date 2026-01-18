from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
def validate_ActiveDirectory(config, label='ActiveDirectory'):
    users_qry = config.get('usersQuery')
    if not users_qry or not isinstance(users_qry, dict):
        return '{0}: usersQuery requires as dictionnary'.format(label)
    error = validate_ldap_query(users_qry)
    if not error:
        return error
    for field in ('userNameAttributes', 'groupMembershipAttributes'):
        value = config.get(field)
        if not value:
            return '{0}: {1} is required.'.format(field, label)
    return None