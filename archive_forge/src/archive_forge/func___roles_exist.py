from __future__ import absolute_import, division, print_function
from datetime import timedelta
from decimal import Decimal
from os import environ
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def __roles_exist(self, roles):
    tmp = ["'" + x + "'" for x in roles]
    query = 'SELECT rolname FROM pg_roles WHERE rolname IN (%s)' % ','.join(tmp)
    return [x['rolname'] for x in exec_sql(self, query, add_to_executed=False)]