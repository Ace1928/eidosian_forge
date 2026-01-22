from __future__ import absolute_import, division, print_function
from decimal import Decimal
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def __get_engines(self):
    """Get storage engines info."""
    res = self.__exec_sql('SHOW ENGINES')
    if res:
        for line in res:
            engine = line['Engine']
            self.info['engines'][engine] = {}
            for vname, val in iteritems(line):
                if vname != 'Engine':
                    self.info['engines'][engine][vname] = val