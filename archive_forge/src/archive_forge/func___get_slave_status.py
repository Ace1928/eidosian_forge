from __future__ import absolute_import, division, print_function
from decimal import Decimal
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def __get_slave_status(self):
    """Get slave status if the instance is a slave."""
    if self.server_implementation == 'mariadb':
        res = self.__exec_sql('SHOW ALL SLAVES STATUS')
    else:
        res = self.__exec_sql('SHOW SLAVE STATUS')
    if res:
        for line in res:
            host = line['Master_Host']
            if host not in self.info['slave_status']:
                self.info['slave_status'][host] = {}
            port = line['Master_Port']
            if port not in self.info['slave_status'][host]:
                self.info['slave_status'][host][port] = {}
            user = line['Master_User']
            if user not in self.info['slave_status'][host][port]:
                self.info['slave_status'][host][port][user] = {}
            for vname, val in iteritems(line):
                if vname not in ('Master_Host', 'Master_Port', 'Master_User'):
                    self.info['slave_status'][host][port][user][vname] = self.__convert(val)