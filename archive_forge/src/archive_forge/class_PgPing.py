from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
class PgPing(object):

    def __init__(self, module, cursor):
        self.module = module
        self.cursor = cursor
        self.is_available = False
        self.version = {}

    def do(self):
        self.get_pg_version()
        return (self.is_available, self.version)

    def get_pg_version(self):
        query = 'SELECT version()'
        raw = exec_sql(self, query, add_to_executed=False)[0]['version']
        if not raw:
            return
        self.is_available = True
        full = raw.split()[1]
        m = re.match('(\\d+)\\.(\\d+)(?:\\.(\\d+))?', full)
        major = int(m.group(1))
        minor = int(m.group(2))
        patch = None
        if m.group(3) is not None:
            patch = int(m.group(3))
        self.version = dict(major=major, minor=minor, full=full, raw=raw)
        if patch is not None:
            self.version['patch'] = patch