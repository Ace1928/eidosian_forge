from __future__ import absolute_import, division, print_function
from decimal import Decimal
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_native
class MySQL_Info(object):
    """Class for collection MySQL instance information.

    Arguments:
        module (AnsibleModule): Object of AnsibleModule class.
        cursor (pymysql/mysql-python): Cursor class for interaction with
            the database.

    Note:
        If you need to add a new subset:
        1. add a new key with the same name to self.info attr in self.__init__()
        2. add a new private method to get the information
        3. add invocation of the new method to self.__collect()
        4. add info about the new subset to the DOCUMENTATION block
        5. add info about the new subset with an example to RETURN block
    """

    def __init__(self, module, cursor, server_implementation):
        self.module = module
        self.cursor = cursor
        self.server_implementation = server_implementation
        self.info = {'version': {}, 'databases': {}, 'settings': {}, 'global_status': {}, 'engines': {}, 'users': {}, 'users_info': {}, 'master_status': {}, 'slave_hosts': {}, 'slave_status': {}}

    def get_info(self, filter_, exclude_fields, return_empty_dbs):
        """Get MySQL instance information based on filter_.

        Arguments:
            filter_ (list): List of collected subsets (e.g., databases, users, etc.),
                when it is empty, return all available information.
        """
        inc_list = []
        exc_list = []
        if filter_:
            partial_info = {}
            for fi in filter_:
                if fi.lstrip('!') not in self.info:
                    self.module.warn('filter element: %s is not allowable, ignored' % fi)
                    continue
                if fi[0] == '!':
                    exc_list.append(fi.lstrip('!'))
                else:
                    inc_list.append(fi)
            if inc_list:
                self.__collect(exclude_fields, return_empty_dbs, set(inc_list))
                for i in self.info:
                    if i in inc_list:
                        partial_info[i] = self.info[i]
            else:
                not_in_exc_list = list(set(self.info) - set(exc_list))
                self.__collect(exclude_fields, return_empty_dbs, set(not_in_exc_list))
                for i in self.info:
                    if i not in exc_list:
                        partial_info[i] = self.info[i]
            return partial_info
        else:
            self.__collect(exclude_fields, return_empty_dbs, set(self.info))
            return self.info

    def __collect(self, exclude_fields, return_empty_dbs, wanted):
        """Collect all possible subsets."""
        if 'version' in wanted or 'settings' in wanted:
            self.__get_global_variables()
        if 'databases' in wanted:
            self.__get_databases(exclude_fields, return_empty_dbs)
        if 'global_status' in wanted:
            self.__get_global_status()
        if 'engines' in wanted:
            self.__get_engines()
        if 'users' in wanted:
            self.__get_users()
        if 'users_info' in wanted:
            self.__get_users_info()
        if 'master_status' in wanted:
            self.__get_master_status()
        if 'slave_status' in wanted:
            self.__get_slave_status()
        if 'slave_hosts' in wanted:
            self.__get_slaves()

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

    def __convert(self, val):
        """Convert unserializable data."""
        try:
            if isinstance(val, Decimal):
                val = float(val)
            else:
                val = int(val)
        except ValueError:
            pass
        except TypeError:
            pass
        return val

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

    def __get_global_status(self):
        """Get global status."""
        res = self.__exec_sql('SHOW GLOBAL STATUS')
        if res:
            for var in res:
                self.info['global_status'][var['Variable_name']] = self.__convert(var['Value'])

    def __get_master_status(self):
        """Get master status if the instance is a master."""
        res = self.__exec_sql('SHOW MASTER STATUS')
        if res:
            for line in res:
                for vname, val in iteritems(line):
                    self.info['master_status'][vname] = self.__convert(val)

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

    def __get_slaves(self):
        """Get slave hosts info if the instance is a master."""
        res = self.__exec_sql('SHOW SLAVE HOSTS')
        if res:
            for line in res:
                srv_id = line['Server_id']
                if srv_id not in self.info['slave_hosts']:
                    self.info['slave_hosts'][srv_id] = {}
                for vname, val in iteritems(line):
                    if vname != 'Server_id':
                        self.info['slave_hosts'][srv_id][vname] = self.__convert(val)

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

    def __get_users_info(self):
        """Get user privileges, passwords, resources_limits, ...

        Query the server to get all the users and return a string
        of privileges that can be used by the mysql_user plugin.
        For instance:

        "users_info": [
            {
                "host": "users_info.com",
                "priv": "*.*: ALL,GRANT",
                "name": "users_info_adm"
            },
            {
                "host": "users_info.com",
                "priv": "`mysql`.*: SELECT/`users_info_db`.*: SELECT",
                "name": "users_info_multi"
            }
        ]
        """
        res = self.__exec_sql('SELECT * FROM mysql.user')
        if not res:
            return None
        output = list()
        for line in res:
            user = line['User']
            host = line['Host']
            user_priv = privileges_get(self.cursor, user, host)
            if not user_priv:
                self.module.warn('No privileges found for %s on host %s' % (user, host))
                continue
            priv_string = list()
            for db_table, priv in user_priv.items():
                if set(priv) == {'PROXY', 'GRANT'} or set(priv) == {'PROXY'}:
                    continue
                unquote_db_table = db_table.replace('`', '').replace("'", '')
                priv_string.append('%s:%s' % (unquote_db_table, ','.join(priv)))
            if len(priv_string) > 1 and '*.*:USAGE' in priv_string:
                priv_string.remove('*.*:USAGE')
            resource_limits = get_resource_limits(self.cursor, user, host)
            copy_ressource_limits = dict.copy(resource_limits)
            output_dict = {'name': user, 'host': host, 'priv': '/'.join(priv_string), 'resource_limits': copy_ressource_limits}
            if resource_limits:
                for key, value in resource_limits.items():
                    if value == 0:
                        del output_dict['resource_limits'][key]
                if len(output_dict['resource_limits']) == 0:
                    del output_dict['resource_limits']
            authentications = get_existing_authentication(self.cursor, user, host)
            if authentications:
                output_dict.update(authentications)
            output.append(output_dict)
        self.info['users_info'] = output

    def __get_databases(self, exclude_fields, return_empty_dbs):
        """Get info about databases."""
        if not exclude_fields:
            query = 'SELECT table_schema AS "name", SUM(data_length + index_length) AS "size" FROM information_schema.TABLES GROUP BY table_schema'
        elif 'db_size' in exclude_fields:
            query = 'SELECT table_schema AS "name" FROM information_schema.TABLES GROUP BY table_schema'
        res = self.__exec_sql(query)
        if res:
            for db in res:
                self.info['databases'][db['name']] = {}
                if not exclude_fields or 'db_size' not in exclude_fields:
                    if db['size'] is None:
                        db['size'] = 0
                    self.info['databases'][db['name']]['size'] = int(db['size'])
        if not return_empty_dbs:
            return None
        res = self.__exec_sql('SHOW DATABASES')
        if res:
            for db in res:
                if db['Database'] not in self.info['databases']:
                    self.info['databases'][db['Database']] = {}
                    if not exclude_fields or 'db_size' not in exclude_fields:
                        self.info['databases'][db['Database']]['size'] = 0

    def __exec_sql(self, query, ddl=False):
        """Execute SQL.

        Arguments:
            ddl (bool): If True, return True or False.
                Used for queries that don't return any rows
                (mainly for DDL queries) (default False).
        """
        try:
            self.cursor.execute(query)
            if not ddl:
                res = self.cursor.fetchall()
                return res
            return True
        except Exception as e:
            self.module.fail_json(msg="Cannot execute SQL '%s': %s" % (query, to_native(e)))
        return False