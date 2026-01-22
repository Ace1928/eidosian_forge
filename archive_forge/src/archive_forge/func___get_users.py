from __future__ import absolute_import, division, print_function
from decimal import Decimal
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def __get_users(self):
    """Get user info."""
    res = self.__exec_sql('SELECT * FROM mysql.user')
    if res:
        for line in res:
            host = line['Host']
            if host not in self.info['users']:
                self.info['users'][host] = {}
            user = line['User']
            self.info['users'][host][user] = {}
            for vname, val in iteritems(line):
                if vname not in ('Host', 'User'):
                    self.info['users'][host][user][vname] = self.__convert(val)