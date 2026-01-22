from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.proxysql.plugins.module_utils.mysql import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
class ProxySQLServer(object):

    def __init__(self, module):
        self.state = module.params['state']
        self.save_to_disk = module.params['save_to_disk']
        self.load_to_runtime = module.params['load_to_runtime']
        self.hostgroup_id = module.params['hostgroup_id']
        self.hostname = module.params['hostname']
        self.port = module.params['port']
        config_data_keys = ['status', 'weight', 'compression', 'max_connections', 'max_replication_lag', 'use_ssl', 'max_latency_ms', 'comment']
        self.config_data = dict(((k, module.params[k]) for k in config_data_keys))

    def check_server_config_exists(self, cursor):
        query_string = 'SELECT count(*) AS `host_count`\n               FROM mysql_servers\n               WHERE hostgroup_id = %s\n                 AND hostname = %s\n                 AND port = %s'
        query_data = [self.hostgroup_id, self.hostname, self.port]
        cursor.execute(query_string, query_data)
        check_count = cursor.fetchone()
        return int(check_count['host_count']) > 0

    def check_server_config(self, cursor):
        query_string = 'SELECT count(*) AS `host_count`\n               FROM mysql_servers\n               WHERE hostgroup_id = %s\n                 AND hostname = %s\n                 AND port = %s'
        query_data = [self.hostgroup_id, self.hostname, self.port]
        for col, val in iteritems(self.config_data):
            if val is not None:
                query_data.append(val)
                query_string += '\n  AND ' + col + ' = %s'
        cursor.execute(query_string, query_data)
        check_count = cursor.fetchone()
        if isinstance(check_count, tuple):
            return int(check_count[0]) > 0
        return int(check_count['host_count']) > 0

    def get_server_config(self, cursor):
        query_string = 'SELECT *\n               FROM mysql_servers\n               WHERE hostgroup_id = %s\n                 AND hostname = %s\n                 AND port = %s'
        query_data = [self.hostgroup_id, self.hostname, self.port]
        cursor.execute(query_string, query_data)
        server = cursor.fetchone()
        return server

    def create_server_config(self, cursor):
        query_string = 'INSERT INTO mysql_servers (\n               hostgroup_id,\n               hostname,\n               port'
        cols = 3
        query_data = [self.hostgroup_id, self.hostname, self.port]
        for col, val in iteritems(self.config_data):
            if val is not None:
                cols += 1
                query_data.append(val)
                query_string += ',\n' + col
        query_string += ')\n' + 'VALUES (' + '%s ,' * cols
        query_string = query_string[:-2]
        query_string += ')'
        cursor.execute(query_string, query_data)
        return True

    def update_server_config(self, cursor):
        query_string = 'UPDATE mysql_servers'
        cols = 0
        query_data = []
        for col, val in iteritems(self.config_data):
            if val is not None:
                cols += 1
                query_data.append(val)
                if cols == 1:
                    query_string += '\nSET ' + col + '= %s,'
                else:
                    query_string += '\n    ' + col + ' = %s,'
        query_string = query_string[:-1]
        query_string += '\nWHERE hostgroup_id = %s\n  AND hostname = %s' + '\n  AND port = %s'
        query_data.append(self.hostgroup_id)
        query_data.append(self.hostname)
        query_data.append(self.port)
        cursor.execute(query_string, query_data)
        return True

    def delete_server_config(self, cursor):
        query_string = 'DELETE FROM mysql_servers\n               WHERE hostgroup_id = %s\n                 AND hostname = %s\n                 AND port = %s'
        query_data = [self.hostgroup_id, self.hostname, self.port]
        cursor.execute(query_string, query_data)
        return True

    def manage_config(self, cursor, state):
        if state:
            if self.save_to_disk:
                save_config_to_disk(cursor, 'SERVERS')
            if self.load_to_runtime:
                load_config_to_runtime(cursor, 'SERVERS')

    def create_server(self, check_mode, result, cursor):
        if not check_mode:
            result['changed'] = self.create_server_config(cursor)
            result['msg'] = 'Added server to mysql_hosts'
            result['server'] = self.get_server_config(cursor)
            self.manage_config(cursor, result['changed'])
        else:
            result['changed'] = True
            result['msg'] = 'Server would have been added to' + ' mysql_hosts, however check_mode' + ' is enabled.'

    def update_server(self, check_mode, result, cursor):
        if not check_mode:
            result['changed'] = self.update_server_config(cursor)
            result['msg'] = 'Updated server in mysql_hosts'
            result['server'] = self.get_server_config(cursor)
            self.manage_config(cursor, result['changed'])
        else:
            result['changed'] = True
            result['msg'] = 'Server would have been updated in' + ' mysql_hosts, however check_mode' + ' is enabled.'

    def delete_server(self, check_mode, result, cursor):
        if not check_mode:
            result['server'] = self.get_server_config(cursor)
            result['changed'] = self.delete_server_config(cursor)
            result['msg'] = 'Deleted server from mysql_hosts'
            self.manage_config(cursor, result['changed'])
        else:
            result['changed'] = True
            result['msg'] = 'Server would have been deleted from' + ' mysql_hosts, however check_mode is' + ' enabled.'