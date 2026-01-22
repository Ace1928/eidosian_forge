from __future__ import absolute_import, division, print_function
import re
from fnmatch import fnmatch
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
class PgClusterInfo(object):
    """Class for collection information about a PostgreSQL instance.

    Arguments:
        module (AnsibleModule): Object of AnsibleModule class.
        db_conn_obj (psycopg.connect): PostgreSQL connection object.
    """

    def __init__(self, module, db_conn_obj):
        self.module = module
        self.db_obj = db_conn_obj
        self.cursor = db_conn_obj.connect()
        self.pg_info = {'version': {}, 'in_recovery': None, 'tablespaces': {}, 'databases': {}, 'replications': {}, 'repl_slots': {}, 'settings': {}, 'roles': {}, 'pending_restart_settings': []}

    def collect(self, val_list=False):
        """Collect information based on 'filter' option."""
        subset_map = {'version': self.get_pg_version, 'in_recovery': self.get_recovery_state, 'tablespaces': self.get_tablespaces, 'databases': self.get_db_info, 'replications': self.get_repl_info, 'repl_slots': self.get_rslot_info, 'settings': self.get_settings, 'roles': self.get_role_info}
        incl_list = []
        excl_list = []
        if val_list:
            for i in val_list:
                if i[0] != '!':
                    incl_list.append(i)
                else:
                    excl_list.append(i.lstrip('!'))
            if incl_list:
                for s in subset_map:
                    for i in incl_list:
                        if fnmatch(s, i):
                            subset_map[s]()
                            break
            elif excl_list:
                found = False
                for s in subset_map:
                    for e in excl_list:
                        if fnmatch(s, e):
                            found = True
                    if not found:
                        subset_map[s]()
                    else:
                        found = False
        else:
            for s in subset_map:
                subset_map[s]()
        self.cursor.close()
        self.db_obj.db_conn.close()
        return self.pg_info

    def get_pub_info(self):
        """Get publication statistics."""
        query = 'SELECT p.*, r.rolname AS ownername FROM pg_catalog.pg_publication AS p JOIN pg_catalog.pg_roles AS r ON p.pubowner = r.oid'
        result = self.__exec_sql(query)
        if result:
            result = [dict(row) for row in result]
        else:
            return {}
        publications = {}
        for elem in result:
            if not publications.get(elem['pubname']):
                publications[elem['pubname']] = {}
            for key, val in iteritems(elem):
                if key != 'pubname':
                    publications[elem['pubname']][key] = val
        return publications

    def get_subscr_info(self):
        """Get subscription statistics."""
        columns_sub_table = "SELECT column_name FROM information_schema.columns WHERE table_schema = 'pg_catalog' AND table_name = 'pg_subscription'"
        columns_result = self.__exec_sql(columns_sub_table)
        columns = ', '.join(['s.%s' % column['column_name'] for column in columns_result])
        query = 'SELECT %s, r.rolname AS ownername, d.datname AS dbname FROM pg_catalog.pg_subscription s JOIN pg_catalog.pg_database d ON s.subdbid = d.oid JOIN pg_catalog.pg_roles AS r ON s.subowner = r.oid' % columns
        result = self.__exec_sql(query)
        if result:
            result = [dict(row) for row in result]
        else:
            return {}
        subscr_info = {}
        for elem in result:
            if not subscr_info.get(elem['dbname']):
                subscr_info[elem['dbname']] = {}
            if not subscr_info[elem['dbname']].get(elem['subname']):
                subscr_info[elem['dbname']][elem['subname']] = {}
                for key, val in iteritems(elem):
                    if key not in ('subname', 'dbname'):
                        subscr_info[elem['dbname']][elem['subname']][key] = val
        return subscr_info

    def get_tablespaces(self):
        """Get information about tablespaces."""
        opt = self.__exec_sql("SELECT column_name FROM information_schema.columns WHERE table_name = 'pg_tablespace' AND column_name = 'spcoptions'")
        if not opt:
            query = 'SELECT s.spcname, pg_catalog.pg_get_userbyid(s.spcowner) as rolname, s.spcacl::text FROM pg_tablespace AS s '
        else:
            query = 'SELECT s.spcname, pg_catalog.pg_get_userbyid(s.spcowner) as rolname, s.spcacl::text, s.spcoptions FROM pg_tablespace AS s '
        res = self.__exec_sql(query)
        ts_dict = {}
        for i in res:
            ts_name = i['spcname']
            ts_info = dict(spcowner=i['rolname'], spcacl=i['spcacl'] if i['spcacl'] else '')
            if opt:
                ts_info['spcoptions'] = i['spcoptions'] if i['spcoptions'] else []
            ts_dict[ts_name] = ts_info
        self.pg_info['tablespaces'] = ts_dict

    def get_ext_info(self):
        """Get information about existing extensions."""
        res = self.__exec_sql("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'pg_extension')")
        if not res[0]['exists']:
            return True
        query = "SELECT e.extname, e.extversion, n.nspname, c.description FROM pg_catalog.pg_extension AS e LEFT JOIN pg_catalog.pg_namespace AS n ON n.oid = e.extnamespace LEFT JOIN pg_catalog.pg_description AS c ON c.objoid = e.oid AND c.classoid = 'pg_catalog.pg_extension'::pg_catalog.regclass"
        res = self.__exec_sql(query)
        ext_dict = {}
        for i in res:
            ext_ver_raw = i['extversion']
            if re.search('^([0-9]+([\\-]*[0-9]+)?\\.)*[0-9]+([\\-]*[0-9]+)?$', i['extversion']) is None:
                ext_ver = [None, None]
            else:
                ext_ver = i['extversion'].split('.')
                if re.search('-', ext_ver[0]) is not None:
                    ext_ver = ext_ver[0].split('-')
                else:
                    try:
                        if re.search('-', ext_ver[1]) is not None:
                            ext_ver[1] = ext_ver[1].split('-')[0]
                    except IndexError:
                        ext_ver.append(None)
            ext_dict[i['extname']] = dict(extversion=dict(major=int(ext_ver[0]) if ext_ver[0] else None, minor=int(ext_ver[1]) if ext_ver[1] else None, raw=ext_ver_raw), nspname=i['nspname'], description=i['description'])
        return ext_dict

    def get_role_info(self):
        """Get information about roles (in PgSQL groups and users are roles)."""
        query = "SELECT r.rolname, r.rolsuper, r.rolcanlogin, r.rolvaliduntil, ARRAY(SELECT b.rolname FROM pg_catalog.pg_auth_members AS m JOIN pg_catalog.pg_roles AS b ON (m.roleid = b.oid) WHERE m.member = r.oid) AS memberof FROM pg_catalog.pg_roles AS r WHERE r.rolname !~ '^pg_'"
        res = self.__exec_sql(query)
        rol_dict = {}
        for i in res:
            rol_dict[i['rolname']] = dict(superuser=i['rolsuper'], canlogin=i['rolcanlogin'], valid_until=i['rolvaliduntil'] if i['rolvaliduntil'] else '', member_of=i['memberof'] if i['memberof'] else [])
        self.pg_info['roles'] = rol_dict

    def get_rslot_info(self):
        """Get information about replication slots if exist."""
        res = self.__exec_sql("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'pg_replication_slots')")
        if not res[0]['exists']:
            return True
        query = 'SELECT slot_name, plugin, slot_type, database, active FROM pg_replication_slots'
        res = self.__exec_sql(query)
        if not res:
            return True
        rslot_dict = {}
        for i in res:
            rslot_dict[i['slot_name']] = dict(plugin=i['plugin'], slot_type=i['slot_type'], database=i['database'], active=i['active'])
        self.pg_info['repl_slots'] = rslot_dict

    def get_settings(self):
        """Get server settings."""
        pend_rest_col_exists = self.__exec_sql("SELECT 1 FROM information_schema.columns WHERE table_name = 'pg_settings' AND column_name = 'pending_restart'")
        if not pend_rest_col_exists:
            query = 'SELECT name, setting, unit, context, vartype, boot_val, min_val, max_val, sourcefile FROM pg_settings'
        else:
            query = 'SELECT name, setting, unit, context, vartype, boot_val, min_val, max_val, sourcefile, pending_restart FROM pg_settings'
        res = self.__exec_sql(query)
        set_dict = {}
        for i in res:
            val_in_bytes = None
            setting = i['setting']
            if i['unit']:
                unit = i['unit']
            else:
                unit = ''
            if unit == 'kB':
                val_in_bytes = int(setting) * 1024
            elif unit == '8kB':
                val_in_bytes = int(setting) * 1024 * 8
            elif unit == 'MB':
                val_in_bytes = int(setting) * 1024 * 1024
            if val_in_bytes is not None and val_in_bytes < 0:
                val_in_bytes = 0
            setting_name = i['name']
            pretty_val = self.__get_pretty_val(setting_name)
            pending_restart = None
            if pend_rest_col_exists:
                pending_restart = i['pending_restart']
            set_dict[setting_name] = dict(setting=setting, unit=unit, context=i['context'], vartype=i['vartype'], boot_val=i['boot_val'] if i['boot_val'] else '', min_val=i['min_val'] if i['min_val'] else '', max_val=i['max_val'] if i['max_val'] else '', sourcefile=i['sourcefile'] if i['sourcefile'] else '', pretty_val=pretty_val)
            if val_in_bytes is not None:
                set_dict[setting_name]['val_in_bytes'] = val_in_bytes
            if pending_restart is not None:
                set_dict[setting_name]['pending_restart'] = pending_restart
                if pending_restart:
                    self.pg_info['pending_restart_settings'].append(setting_name)
        self.pg_info['settings'] = set_dict

    def get_repl_info(self):
        """Get information about replication if the server is a primary."""
        res = self.__exec_sql("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'pg_stat_replication')")
        if not res[0]['exists']:
            return True
        query = 'SELECT r.pid, pg_catalog.pg_get_userbyid(r.usesysid) AS rolname, r.application_name, r.client_addr::text, r.client_hostname, r.backend_start::text, r.state FROM pg_stat_replication AS r '
        res = self.__exec_sql(query)
        if not res:
            return True
        repl_dict = {}
        for i in res:
            repl_dict[i['pid']] = dict(usename=i['rolname'], app_name=i['application_name'] if i['application_name'] else '', client_addr=i['client_addr'], client_hostname=i['client_hostname'] if i['client_hostname'] else '', backend_start=i['backend_start'], state=i['state'])
        self.pg_info['replications'] = repl_dict

    def get_lang_info(self):
        """Get information about current supported languages."""
        query = 'SELECT l.lanname, pg_catalog.pg_get_userbyid(l.lanowner) AS rolname, l.lanacl::text FROM pg_language AS l '
        res = self.__exec_sql(query)
        lang_dict = {}
        for i in res:
            lang_dict[i['lanname']] = dict(lanowner=i['rolname'], lanacl=i['lanacl'] if i['lanacl'] else '')
        return lang_dict

    def get_namespaces(self):
        """Get information about namespaces."""
        query = 'SELECT n.nspname, pg_catalog.pg_get_userbyid(n.nspowner) AS rolname, n.nspacl::text FROM pg_catalog.pg_namespace AS n '
        res = self.__exec_sql(query)
        nsp_dict = {}
        for i in res:
            nsp_dict[i['nspname']] = dict(nspowner=i['rolname'], nspacl=i['nspacl'] if i['nspacl'] else '')
        return nsp_dict

    def get_pg_version(self):
        """Get major and minor PostgreSQL server version."""
        query = 'SELECT version()'
        raw = self.__exec_sql(query)[0]['version']
        full = raw.split()[1]
        m = re.match('(\\d+)\\.(\\d+)(?:\\.(\\d+))?', full)
        major = int(m.group(1))
        minor = int(m.group(2))
        patch = None
        if m.group(3) is not None:
            patch = int(m.group(3))
        self.pg_info['version'] = dict(major=major, minor=minor, full=full, raw=raw)
        if patch is not None:
            self.pg_info['version']['patch'] = patch

    def get_recovery_state(self):
        """Get if the service is in recovery mode."""
        self.pg_info['in_recovery'] = self.__exec_sql('SELECT pg_is_in_recovery()')[0]['pg_is_in_recovery']

    def get_db_info(self):
        """Get information about the current database."""
        query = "SELECT d.datname, pg_catalog.pg_get_userbyid(d.datdba) AS username, pg_catalog.pg_encoding_to_char(d.encoding) AS encoding, d.datcollate, d.datctype, pg_catalog.array_to_string(d.datacl, E'\n') aclstring, CASE WHEN pg_catalog.has_database_privilege(d.datname, 'CONNECT') THEN pg_catalog.pg_database_size(d.datname)::text ELSE 'No Access' END as dbsize, t.spcname FROM pg_catalog.pg_database AS d JOIN pg_catalog.pg_tablespace t ON d.dattablespace = t.oid WHERE d.datname != 'template0'"
        res = self.__exec_sql(query)
        db_dict = {}
        for i in res:
            db_dict[i['datname']] = dict(owner=i['username'], encoding=i['encoding'], collate=i['datcollate'], ctype=i['datctype'], access_priv=i['aclstring'] if i['aclstring'] else '', size=i['dbsize'])
        if get_server_version(self.cursor.connection) >= 100000:
            subscr_info = self.get_subscr_info()
        for datname in db_dict:
            self.cursor = self.db_obj.reconnect(datname)
            if self.cursor is None:
                db_dict[datname]['namespaces'] = {}
                db_dict[datname]['extensions'] = {}
                db_dict[datname]['languages'] = {}
                db_dict[datname]['error'] = 'Could not connect to the database.'
                continue
            db_dict[datname]['namespaces'] = self.get_namespaces()
            db_dict[datname]['extensions'] = self.get_ext_info()
            db_dict[datname]['languages'] = self.get_lang_info()
            if get_server_version(self.cursor.connection) >= 100000:
                db_dict[datname]['publications'] = self.get_pub_info()
                db_dict[datname]['subscriptions'] = subscr_info.get(datname, {})
        self.pg_info['databases'] = db_dict

    def __get_pretty_val(self, setting):
        """Get setting's value represented by SHOW command."""
        return self.__exec_sql('SHOW "%s"' % setting)[0][setting]

    def __exec_sql(self, query):
        """Execute SQL and return the result."""
        try:
            self.cursor.execute(query)
            res = self.cursor.fetchall()
            if res:
                return res
        except Exception as e:
            self.module.fail_json(msg="Cannot execute SQL '%s': %s" % (query, to_native(e)))
            self.cursor.close()
        return False