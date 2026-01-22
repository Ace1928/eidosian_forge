from __future__ import absolute_import, division, print_function
from decimal import Decimal
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
def __get_global_variables(self):
    """Get global variables (instance settings)."""
    res = self.__exec_sql('SHOW GLOBAL VARIABLES')
    if res:
        for var in res:
            self.info['settings'][var['Variable_name']] = self.__convert(var['Value'])
        version = self.info['settings']['version'].split('.')
        full = self.info['settings']['version']
        release = version[2].split('-')[0]
        if len(version[2].split('-')) > 1:
            suffix = version[2].split('-', 1)[1]
        else:
            suffix = ''
        self.info['version'] = dict(major=int(version[0]), minor=int(version[1]), release=int(release), suffix=str(suffix), full=str(full))