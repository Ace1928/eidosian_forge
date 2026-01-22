from __future__ import absolute_import, division, print_function
from decimal import Decimal
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def __get_global_status(self):
    """Get global status."""
    res = self.__exec_sql('SHOW GLOBAL STATUS')
    if res:
        for var in res:
            self.info['global_status'][var['Variable_name']] = self.__convert(var['Value'])