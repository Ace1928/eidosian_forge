from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
def validate_RFC2307(config):
    qry = config.get('groupsQuery')
    if not qry or not isinstance(qry, dict):
        return 'RFC2307: groupsQuery requires a dictionary'
    error = validate_ldap_query(qry)
    if not error:
        return error
    for field in ('groupUIDAttribute', 'groupNameAttributes', 'groupMembershipAttributes', 'userUIDAttribute', 'userNameAttributes'):
        value = config.get(field)
        if not value:
            return 'RFC2307: {0} is required.'.format(field)
    users_qry = config.get('usersQuery')
    if not users_qry or not isinstance(users_qry, dict):
        return 'RFC2307: usersQuery requires a dictionary'
    isUserDNOnly = config.get('userUIDAttribute').strip() == 'dn'
    return validate_ldap_query(users_qry, isDNOnly=isUserDNOnly)