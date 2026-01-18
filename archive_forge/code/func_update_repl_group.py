from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils._text import to_native
def update_repl_group(self, result, cursor):
    current = self.get_repl_group_config(cursor)
    if self.check_type_support and current.get('check_type') != self.check_type:
        result['changed'] = True
        if not self.check_mode:
            result['msg'] = 'Updated replication hostgroups'
            self.update_check_type(cursor)
        else:
            result['msg'] = 'Updated replication hostgroups in check_mode'
    if current.get('comment') != self.comment:
        result['changed'] = True
        result['msg'] = 'Updated replication hostgroups in check_mode'
        if not self.check_mode:
            result['msg'] = 'Updated replication hostgroups'
            self.update_comment(cursor)
    if int(current.get('reader_hostgroup')) != self.reader_hostgroup:
        result['changed'] = True
        result['msg'] = 'Updated replication hostgroups in check_mode'
        if not self.check_mode:
            result['msg'] = 'Updated replication hostgroups'
            self.update_reader_hostgroup(cursor)
    result['repl_group'] = self.get_repl_group_config(cursor)
    self.manage_config(cursor, result['changed'])