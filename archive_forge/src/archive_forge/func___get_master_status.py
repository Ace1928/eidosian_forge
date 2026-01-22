from __future__ import absolute_import, division, print_function
from decimal import Decimal
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def __get_master_status(self):
    """Get master status if the instance is a master."""
    res = self.__exec_sql('SHOW MASTER STATUS')
    if res:
        for line in res:
            for vname, val in iteritems(line):
                self.info['master_status'][vname] = self.__convert(val)