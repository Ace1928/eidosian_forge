from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
from ansible_collections.community.postgresql.plugins.module_utils.version import \
def add_grant_option(self):
    if self._grant_option:
        if self._obj_type == 'group':
            self.query[-1] += ' WITH ADMIN OPTION;'
        else:
            self.query[-1] += ' WITH GRANT OPTION;'
    elif self._grant_option is False:
        self.query[-1] += ';'
        if self._obj_type == 'group':
            self.query.append('REVOKE ADMIN OPTION FOR {0} FROM {1};'.format(self._set_what, self._for_whom))
        elif not self._obj_type == 'default_privs':
            self.query.append('REVOKE GRANT OPTION FOR {0} FROM {1};'.format(self._set_what, self._for_whom))
    else:
        self.query[-1] += ';'