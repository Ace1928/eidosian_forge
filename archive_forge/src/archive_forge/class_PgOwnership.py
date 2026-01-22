from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
class PgOwnership(object):
    """Class for changing ownership of PostgreSQL objects.

    Arguments:
        module (AnsibleModule): Object of Ansible module class.
        cursor (psycopg.connect.cursor): Cursor object for interaction with the database.
        role (str): Role name to set as a new owner of objects.

    Important:
        If you want to add handling of a new type of database objects:
        1. Add a specific method for this like self.__set_db_owner(), etc.
        2. Add a condition with a check of ownership for new type objects to self.__is_owner()
        3. Add a condition with invocation of the specific method to self.set_owner()
        4. Add the information to the module documentation
        That's all.
    """

    def __init__(self, module, cursor, pg_version, role):
        self.module = module
        self.cursor = cursor
        self.pg_version = pg_version
        self.check_role_exists(role)
        self.role = role
        self.changed = False
        self.executed_queries = []
        self.obj_name = ''
        self.obj_type = ''

    def check_role_exists(self, role, fail_on_role=True):
        """Check the role exists or not.

        Arguments:
            role (str): Role name.
            fail_on_role (bool): If True, fail when the role does not exist.
                Otherwise just warn and continue.
        """
        if not self.__role_exists(role):
            if fail_on_role:
                self.module.fail_json(msg="Role '%s' does not exist" % role)
            else:
                self.module.warn("Role '%s' does not exist, pass" % role)
            return False
        else:
            return True

    def reassign(self, old_owners, fail_on_role):
        """Implements REASSIGN OWNED BY command.

        If success, set self.changed as True.

        Arguments:
            old_owners (list): The ownership of all the objects within
                the current database, and of all shared objects (databases, tablespaces),
                owned by these roles will be reassigned to self.role.
            fail_on_role (bool): If True, fail when a role from old_owners does not exist.
                Otherwise just warn and continue.
        """
        roles = []
        for r in old_owners:
            if self.check_role_exists(r, fail_on_role):
                roles.append('"%s"' % r)
        if not roles:
            return False
        old_owners = ','.join(roles)
        query = ['REASSIGN OWNED BY']
        query.append(old_owners)
        query.append('TO "%s"' % self.role)
        query = ' '.join(query)
        self.changed = exec_sql(self, query, return_bool=True)

    def set_owner(self, obj_type, obj_name):
        """Change owner of a database object.

        Arguments:
            obj_type (str): Type of object (like database, table, view, etc.).
            obj_name (str): Object name.
        """
        self.obj_name = obj_name
        self.obj_type = obj_type
        if self.__is_owner():
            return False
        if obj_type == 'database':
            self.__set_db_owner()
        elif obj_type == 'function':
            self.__set_func_owner()
        elif obj_type == 'sequence':
            self.__set_seq_owner()
        elif obj_type == 'schema':
            self.__set_schema_owner()
        elif obj_type == 'table':
            self.__set_table_owner()
        elif obj_type == 'tablespace':
            self.__set_tablespace_owner()
        elif obj_type == 'view':
            self.__set_view_owner()
        elif obj_type == 'matview':
            self.__set_mat_view_owner()
        elif obj_type == 'procedure':
            self.__set_procedure_owner()
        elif obj_type == 'type':
            self.__set_type_owner()
        elif obj_type == 'aggregate':
            self.__set_aggregate_owner()
        elif obj_type == 'routine':
            self.__set_routine_owner()
        elif obj_type == 'language':
            self.__set_language_owner()
        elif obj_type == 'domain':
            self.__set_domain_owner()
        elif obj_type == 'collation':
            self.__set_collation_owner()
        elif obj_type == 'conversion':
            self.__set_conversion_owner()
        elif obj_type == 'text_search_configuration':
            self.__set_text_search_configuration_owner()
        elif obj_type == 'text_search_dictionary':
            self.__set_text_search_dictionary_owner()
        elif obj_type == 'foreign_data_wrapper':
            self.__set_foreign_data_wrapper_owner()
        elif obj_type == 'server':
            self.__set_server_owner()
        elif obj_type == 'foreign_table':
            self.__set_foreign_table_owner()
        elif obj_type == 'event_trigger':
            self.__set_event_trigger_owner()
        elif obj_type == 'large_object':
            self.__set_large_object_owner()
        elif obj_type == 'publication':
            self.__set_publication_owner()
        elif obj_type == 'statistics':
            self.__set_statistics_owner()

    def __is_owner(self):
        """Return True if self.role is the current object owner."""
        if self.obj_type == 'table':
            query = 'SELECT 1 FROM pg_tables WHERE tablename = %(obj_name)s AND tableowner = %(role)s'
        elif self.obj_type == 'database':
            query = 'SELECT 1 FROM pg_database AS d JOIN pg_roles AS r ON d.datdba = r.oid WHERE d.datname = %(obj_name)s AND r.rolname = %(role)s'
        elif self.obj_type in ('aggregate', 'function', 'routine', 'procedure'):
            if self.obj_type == 'routine' and self.pg_version < 110000:
                self.module.fail_json(msg='PostgreSQL version must be >= 11 for obj_type=routine.')
            if self.obj_type == 'procedure' and self.pg_version < 110000:
                self.module.fail_json(msg='PostgreSQL version must be >= 11 for obj_type=procedure.')
            query = 'SELECT 1 FROM pg_proc AS f JOIN pg_roles AS r ON f.proowner = r.oid WHERE f.proname = %(obj_name)s AND r.rolname = %(role)s'
        elif self.obj_type == 'sequence':
            query = "SELECT 1 FROM pg_class AS c JOIN pg_roles AS r ON c.relowner = r.oid WHERE c.relkind = 'S' AND c.relname = %(obj_name)s AND r.rolname = %(role)s"
        elif self.obj_type == 'schema':
            query = 'SELECT 1 FROM information_schema.schemata WHERE schema_name = %(obj_name)s AND schema_owner = %(role)s'
        elif self.obj_type == 'tablespace':
            query = 'SELECT 1 FROM pg_tablespace AS t JOIN pg_roles AS r ON t.spcowner = r.oid WHERE t.spcname = %(obj_name)s AND r.rolname = %(role)s'
        elif self.obj_type == 'view':
            query = 'SELECT 1 FROM pg_views WHERE viewname = %(obj_name)s AND viewowner = %(role)s'
        elif self.obj_type == 'matview':
            if self.pg_version < 90300:
                self.module.fail_json(msg='PostgreSQL version must be >= 9.3 for obj_type=matview.')
            query = 'SELECT 1 FROM pg_matviews WHERE matviewname = %(obj_name)s AND matviewowner = %(role)s'
        elif self.obj_type in ('domain', 'type'):
            query = 'SELECT 1 FROM pg_type AS t JOIN pg_roles AS r ON t.typowner = r.oid WHERE t.typname = %(obj_name)s AND r.rolname = %(role)s'
        elif self.obj_type == 'language':
            query = 'SELECT 1 FROM pg_language AS l JOIN pg_roles AS r ON l.lanowner = r.oid WHERE l.lanname = %(obj_name)s AND r.rolname = %(role)s'
        elif self.obj_type == 'collation':
            query = 'SELECT 1 FROM pg_collation AS c JOIN pg_roles AS r ON c.collowner = r.oid WHERE c.collname = %(obj_name)s AND r.rolname = %(role)s'
        elif self.obj_type == 'conversion':
            query = 'SELECT 1 FROM pg_conversion AS c JOIN pg_roles AS r ON c.conowner = r.oid WHERE c.conname = %(obj_name)s AND r.rolname = %(role)s'
        elif self.obj_type == 'text_search_configuration':
            query = 'SELECT 1 FROM pg_ts_config AS t JOIN pg_roles AS r ON t.cfgowner = r.oid WHERE t.cfgname = %(obj_name)s AND r.rolname = %(role)s'
        elif self.obj_type == 'text_search_dictionary':
            query = 'SELECT 1 FROM pg_ts_dict AS t JOIN pg_roles AS r ON t.dictowner = r.oid WHERE t.dictname = %(obj_name)s AND r.rolname = %(role)s'
        elif self.obj_type == 'foreign_data_wrapper':
            query = 'SELECT 1 FROM pg_foreign_data_wrapper AS f JOIN pg_roles AS r ON f.fdwowner = r.oid WHERE f.fdwname = %(obj_name)s AND r.rolname = %(role)s'
        elif self.obj_type == 'server':
            query = 'SELECT 1 FROM pg_foreign_server AS f JOIN pg_roles AS r ON f.srvowner = r.oid WHERE f.srvname = %(obj_name)s AND r.rolname = %(role)s'
        elif self.obj_type == 'foreign_table':
            query = "SELECT 1 FROM pg_class AS c JOIN pg_roles AS r ON c.relowner = r.oid WHERE c.relkind = 'f' AND c.relname = %(obj_name)s AND r.rolname = %(role)s"
        elif self.obj_type == 'event_trigger':
            if self.pg_version < 110000:
                self.module.fail_json(msg='PostgreSQL version must be >= 11 for obj_type=event_trigger.')
            query = 'SELECT 1 FROM pg_event_trigger AS e JOIN pg_roles AS r ON e.evtowner = r.oid WHERE e.evtname = %(obj_name)s AND r.rolname = %(role)s'
        elif self.obj_type == 'large_object':
            query = 'SELECT 1 FROM pg_largeobject_metadata AS l JOIN pg_roles AS r ON l.lomowner = r.oid WHERE l.oid = %(obj_name)s AND r.rolname = %(role)s'
        elif self.obj_type == 'publication':
            if self.pg_version < 110000:
                self.module.fail_json(msg='PostgreSQL version must be >= 11 for obj_type=publication.')
            query = 'SELECT 1 FROM pg_publication AS p JOIN pg_roles AS r ON p.pubowner = r.oid WHERE p.pubname = %(obj_name)s AND r.rolname = %(role)s'
        elif self.obj_type == 'statistics':
            if self.pg_version < 110000:
                self.module.fail_json(msg='PostgreSQL version must be >= 11 for obj_type=statistics.')
            query = 'SELECT 1 FROM pg_statistic_ext AS s JOIN pg_roles AS r ON s.stxowner = r.oid WHERE s.stxname = %(obj_name)s AND r.rolname = %(role)s'
        if self.obj_type in ('function', 'aggregate', 'procedure', 'routine'):
            query_params = {'obj_name': self.obj_name.split('(')[0], 'role': self.role}
        else:
            query_params = {'obj_name': self.obj_name, 'role': self.role}
        return exec_sql(self, query, query_params, add_to_executed=False)

    def __set_db_owner(self):
        """Set the database owner."""
        query = 'ALTER DATABASE "%s" OWNER TO "%s"' % (self.obj_name, self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_func_owner(self):
        """Set the function owner."""
        query = 'ALTER FUNCTION %s OWNER TO "%s"' % (self.obj_name, self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_seq_owner(self):
        """Set the sequence owner."""
        query = 'ALTER SEQUENCE %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'sequence'), self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_schema_owner(self):
        """Set the schema owner."""
        query = 'ALTER SCHEMA %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'schema'), self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_table_owner(self):
        """Set the table owner."""
        query = 'ALTER TABLE %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'table'), self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_tablespace_owner(self):
        """Set the tablespace owner."""
        query = 'ALTER TABLESPACE "%s" OWNER TO "%s"' % (self.obj_name, self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_view_owner(self):
        """Set the view owner."""
        query = 'ALTER VIEW %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'table'), self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_mat_view_owner(self):
        """Set the materialized view owner."""
        if self.pg_version < 90300:
            self.module.fail_json(msg='PostgreSQL version must be >= 9.3 for obj_type=matview.')
        query = 'ALTER MATERIALIZED VIEW %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'table'), self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_procedure_owner(self):
        """Set the procedure owner."""
        if self.pg_version < 110000:
            self.module.fail_json(msg='PostgreSQL version must be >= 11 for obj_type=procedure.')
        query = 'ALTER PROCEDURE %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'table'), self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_type_owner(self):
        """Set the type owner."""
        query = 'ALTER TYPE %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'table'), self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_aggregate_owner(self):
        """Set the aggregate owner."""
        query = 'ALTER AGGREGATE %s OWNER TO "%s"' % (self.obj_name, self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_routine_owner(self):
        """Set the routine owner."""
        if self.pg_version < 110000:
            self.module.fail_json(msg='PostgreSQL version must be >= 11 for obj_type=routine.')
        query = 'ALTER ROUTINE %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'table'), self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_language_owner(self):
        """Set the language owner."""
        query = 'ALTER LANGUAGE %s OWNER TO "%s"' % (self.obj_name, self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_domain_owner(self):
        """Set the domain owner."""
        query = 'ALTER DOMAIN %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'table'), self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_collation_owner(self):
        """Set the collation owner."""
        query = 'ALTER COLLATION %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'table'), self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_conversion_owner(self):
        """Set the conversion owner."""
        query = 'ALTER CONVERSION %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'table'), self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_text_search_configuration_owner(self):
        """Set the text search configuration owner."""
        query = 'ALTER TEXT SEARCH CONFIGURATION %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'table'), self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_text_search_dictionary_owner(self):
        """Set the text search dictionary owner."""
        query = 'ALTER TEXT SEARCH DICTIONARY %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'table'), self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_foreign_data_wrapper_owner(self):
        """Set the foreign data wrapper owner."""
        query = 'ALTER FOREIGN DATA WRAPPER %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'table'), self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_server_owner(self):
        """Set the server owner."""
        query = 'ALTER SERVER %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'table'), self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_foreign_table_owner(self):
        """Set the foreign table owner."""
        query = 'ALTER FOREIGN TABLE %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'table'), self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_event_trigger_owner(self):
        """Set the event trigger owner."""
        query = 'ALTER EVENT TRIGGER %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'table'), self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_large_object_owner(self):
        """Set the large object owner."""
        query = 'ALTER LARGE OBJECT %s OWNER TO "%s"' % (self.obj_name, self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_publication_owner(self):
        """Set the publication owner."""
        if self.pg_version < 110000:
            self.module.fail_json(msg='PostgreSQL version must be >= 11 for obj_type=publication.')
        query = 'ALTER PUBLICATION %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'publication'), self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __set_statistics_owner(self):
        """Set the statistics owner."""
        if self.pg_version < 110000:
            self.module.fail_json(msg='PostgreSQL version must be >= 11 for obj_type=statistics.')
        query = 'ALTER STATISTICS %s OWNER TO "%s"' % (pg_quote_identifier(self.obj_name, 'table'), self.role)
        self.changed = exec_sql(self, query, return_bool=True)

    def __role_exists(self, role):
        """Return True if role exists, otherwise return False."""
        query_params = {'role': role}
        query = 'SELECT 1 FROM pg_roles WHERE rolname = %(role)s'
        return exec_sql(self, query, query_params, add_to_executed=False)