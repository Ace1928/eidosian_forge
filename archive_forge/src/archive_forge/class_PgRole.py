from __future__ import absolute_import, division, print_function
from datetime import timedelta
from decimal import Decimal
from os import environ
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.version import \
class PgRole:

    def __init__(self, module, cursor, name):
        self.module = module
        self.cursor = cursor
        self.name = name
        self.memberof = self.__fetch_members()

    def __fetch_members(self):
        query = 'SELECT ARRAY(SELECT b.rolname FROM pg_catalog.pg_auth_members m JOIN pg_catalog.pg_roles b ON (m.roleid = b.oid) WHERE m.member = r.oid) FROM pg_catalog.pg_roles r WHERE r.rolname = %(dst_role)s'
        res = exec_sql(self, query, query_params={'dst_role': self.name}, add_to_executed=False)
        if res:
            return res[0]['array']
        else:
            return []