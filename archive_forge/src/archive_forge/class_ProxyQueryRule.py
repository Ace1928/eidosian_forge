from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
class ProxyQueryRule(object):

    def __init__(self, module, version):
        self.state = module.params['state']
        self.force_delete = module.params['force_delete']
        self.save_to_disk = module.params['save_to_disk']
        self.load_to_runtime = module.params['load_to_runtime']
        config_data_keys = ['rule_id', 'active', 'username', 'schemaname', 'flagIN', 'client_addr', 'proxy_addr', 'proxy_port', 'digest', 'match_digest', 'match_pattern', 'negate_match_pattern', 're_modifiers', 'flagOUT', 'replace_pattern', 'destination_hostgroup', 'cache_ttl', 'multiplex', 'timeout', 'retries', 'delay', 'next_query_flagIN', 'mirror_flagOUT', 'mirror_hostgroup', 'error_msg', 'OK_msg', 'multiplex', 'log', 'apply', 'comment']
        if version.get('major') >= 2:
            config_data_keys.append('cache_empty_result')
        self.config_data = dict(((k, module.params[k]) for k in config_data_keys))

    def check_rule_pk_exists(self, cursor):
        query_string = 'SELECT count(*) AS `rule_count`\n               FROM mysql_query_rules\n               WHERE rule_id = %s'
        query_data = [self.config_data['rule_id']]
        cursor.execute(query_string, query_data)
        check_count = cursor.fetchone()
        return int(check_count['rule_count']) > 0

    def check_rule_cfg_exists(self, cursor):
        query_string = 'SELECT count(*) AS `rule_count`\n               FROM mysql_query_rules'
        cols = 0
        query_data = []
        for col, val in iteritems(self.config_data):
            if val is not None:
                cols += 1
                query_data.append(val)
                if cols == 1:
                    query_string += '\n WHERE ' + col + ' = %s'
                else:
                    query_string += '\n  AND ' + col + ' = %s'
        if cols > 0:
            cursor.execute(query_string, query_data)
        else:
            cursor.execute(query_string)
        check_count = cursor.fetchone()
        return int(check_count['rule_count'])

    def get_rule_config(self, cursor, created_rule_id=None):
        query_string = 'SELECT *\n               FROM mysql_query_rules'
        if created_rule_id:
            query_data = [created_rule_id]
            query_string += '\nWHERE rule_id = %s'
            cursor.execute(query_string, query_data)
            rule = cursor.fetchone()
        else:
            cols = 0
            query_data = []
            for col, val in iteritems(self.config_data):
                if val is not None:
                    cols += 1
                    query_data.append(val)
                    if cols == 1:
                        query_string += '\n WHERE ' + col + ' = %s'
                    else:
                        query_string += '\n  AND ' + col + ' = %s'
            if cols > 0:
                cursor.execute(query_string, query_data)
            else:
                cursor.execute(query_string)
            rule = cursor.fetchall()
        return rule

    def create_rule_config(self, cursor):
        query_string = 'INSERT INTO mysql_query_rules ('
        cols = 0
        query_data = []
        for col, val in iteritems(self.config_data):
            if val is not None:
                cols += 1
                query_data.append(val)
                query_string += '\n' + col + ','
        query_string = query_string[:-1]
        query_string += ')\n' + 'VALUES (' + '%s ,' * cols
        query_string = query_string[:-2]
        query_string += ')'
        cursor.execute(query_string, query_data)
        new_rule_id = cursor.lastrowid
        return (True, new_rule_id)

    def update_rule_config(self, cursor):
        query_string = 'UPDATE mysql_query_rules'
        cols = 0
        query_data = []
        for col, val in iteritems(self.config_data):
            if val is not None and col != 'rule_id':
                cols += 1
                query_data.append(val)
                if cols == 1:
                    query_string += '\nSET ' + col + '= %s,'
                else:
                    query_string += '\n    ' + col + ' = %s,'
        query_string = query_string[:-1]
        query_string += '\nWHERE rule_id = %s'
        query_data.append(self.config_data['rule_id'])
        cursor.execute(query_string, query_data)
        return True

    def delete_rule_config(self, cursor):
        query_string = 'DELETE FROM mysql_query_rules'
        cols = 0
        query_data = []
        for col, val in iteritems(self.config_data):
            if val is not None:
                cols += 1
                query_data.append(val)
                if cols == 1:
                    query_string += '\n WHERE ' + col + ' = %s'
                else:
                    query_string += '\n  AND ' + col + ' = %s'
        if cols > 0:
            cursor.execute(query_string, query_data)
        else:
            cursor.execute(query_string)
        check_count = cursor.rowcount
        return (True, int(check_count))

    def manage_config(self, cursor, state):
        if state:
            if self.save_to_disk:
                save_config_to_disk(cursor, 'QUERY RULES')
            if self.load_to_runtime:
                load_config_to_runtime(cursor, 'QUERY RULES')

    def create_rule(self, check_mode, result, cursor):
        if not check_mode:
            result['changed'], new_rule_id = self.create_rule_config(cursor)
            result['msg'] = 'Added rule to mysql_query_rules'
            self.manage_config(cursor, result['changed'])
            result['rules'] = self.get_rule_config(cursor, new_rule_id)
        else:
            result['changed'] = True
            result['msg'] = 'Rule would have been added to' + ' mysql_query_rules, however' + ' check_mode is enabled.'

    def update_rule(self, check_mode, result, cursor):
        if not check_mode:
            result['changed'] = self.update_rule_config(cursor)
            result['msg'] = 'Updated rule in mysql_query_rules'
            self.manage_config(cursor, result['changed'])
            result['rules'] = self.get_rule_config(cursor)
        else:
            result['changed'] = True
            result['msg'] = 'Rule would have been updated in' + ' mysql_query_rules, however' + ' check_mode is enabled.'

    def delete_rule(self, check_mode, result, cursor):
        if not check_mode:
            result['rules'] = self.get_rule_config(cursor)
            result['changed'], result['rows_affected'] = self.delete_rule_config(cursor)
            result['msg'] = 'Deleted rule from mysql_query_rules'
            self.manage_config(cursor, result['changed'])
        else:
            result['changed'] = True
            result['msg'] = 'Rule would have been deleted from' + ' mysql_query_rules, however' + ' check_mode is enabled.'