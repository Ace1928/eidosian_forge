from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
class PgUserObjStatInfo:
    """Class to collect information about PostgreSQL user objects.

    Args:
        module (AnsibleModule): Object of AnsibleModule class.
        cursor (cursor): Cursor object of psycopg library to work with PostgreSQL.

    Attributes:
        module (AnsibleModule): Object of AnsibleModule class.
        cursor (cursor): Cursor object of psycopg library to work with PostgreSQL.
        executed_queries (list): List of executed queries.
        info (dict): Statistics dictionary.
        obj_func_mapping (dict): Mapping of object types to corresponding functions.
        schema (str): Name of a schema to restrict stat collecting.
    """

    def __init__(self, module, cursor):
        self.module = module
        self.cursor = cursor
        self.info = {'functions': {}, 'indexes': {}, 'tables': {}}
        self.obj_func_mapping = {'functions': self.get_func_stat, 'indexes': self.get_idx_stat, 'tables': self.get_tbl_stat}
        self.schema = None

    def collect(self, filter_=None, schema=None):
        """Collect statistics information of user objects.

        Kwargs:
            filter_ (list): List of subsets which need to be collected.
            schema (str): Restrict stat collecting by certain schema.

        Returns:
            ``self.info``.
        """
        if schema:
            self.set_schema(schema)
        if filter_:
            for obj_type in filter_:
                obj_type = obj_type.strip()
                obj_func = self.obj_func_mapping.get(obj_type)
                if obj_func is not None:
                    obj_func()
                else:
                    self.module.warn("Unknown filter option '%s'" % obj_type)
        else:
            for obj_func in self.obj_func_mapping.values():
                obj_func()
        return self.info

    def get_func_stat(self):
        """Get function statistics and fill out self.info dictionary."""
        query = 'SELECT * FROM pg_stat_user_functions'
        qp = None
        if self.schema:
            query = 'SELECT * FROM pg_stat_user_functions WHERE schemaname = %s'
            qp = (self.schema,)
        result = exec_sql(self, query, query_params=qp, add_to_executed=False)
        if not result:
            return
        self.__fill_out_info(result, info_key='functions', schema_key='schemaname', name_key='funcname')

    def get_idx_stat(self):
        """Get index statistics and fill out self.info dictionary."""
        query = 'SELECT * FROM pg_stat_user_indexes'
        qp = None
        if self.schema:
            query = 'SELECT * FROM pg_stat_user_indexes WHERE schemaname = %s'
            qp = (self.schema,)
        result = exec_sql(self, query, query_params=qp, add_to_executed=False)
        if not result:
            return
        self.__fill_out_info(result, info_key='indexes', schema_key='schemaname', name_key='indexrelname')

    def get_tbl_stat(self):
        """Get table statistics and fill out self.info dictionary."""
        query = 'SELECT * FROM pg_stat_user_tables'
        qp = None
        if self.schema:
            query = 'SELECT * FROM pg_stat_user_tables WHERE schemaname = %s'
            qp = (self.schema,)
        result = exec_sql(self, query, query_params=qp, add_to_executed=False)
        if not result:
            return
        self.__fill_out_info(result, info_key='tables', schema_key='schemaname', name_key='relname')

    def __fill_out_info(self, result, info_key=None, schema_key=None, name_key=None):
        result = [dict(row) for row in result]
        for elem in result:
            if not self.info[info_key].get(elem[schema_key]):
                self.info[info_key][elem[schema_key]] = {}
            self.info[info_key][elem[schema_key]][elem[name_key]] = {}
            for key, val in iteritems(elem):
                if key not in (schema_key, name_key):
                    self.info[info_key][elem[schema_key]][elem[name_key]][key] = val
            if info_key in ('tables', 'indexes'):
                schemaname = elem[schema_key]
                if self.schema:
                    schemaname = self.schema
                relname = '%s.%s' % (schemaname, elem[name_key])
                result = exec_sql(self, 'SELECT pg_relation_size (%s)', query_params=(relname,), add_to_executed=False)
                self.info[info_key][elem[schema_key]][elem[name_key]]['size'] = result[0]['pg_relation_size']
                if info_key == 'tables':
                    result = exec_sql(self, 'SELECT pg_total_relation_size (%s)', query_params=(relname,), add_to_executed=False)
                    self.info[info_key][elem[schema_key]][elem[name_key]]['total_size'] = result[0]['pg_total_relation_size']

    def set_schema(self, schema):
        """If schema exists, sets self.schema, otherwise fails."""
        query = 'SELECT 1 as schema_exists FROM information_schema.schemata WHERE schema_name = %s'
        result = exec_sql(self, query, query_params=(schema,), add_to_executed=False)
        if result and result[0]['schema_exists']:
            self.schema = schema
        else:
            self.module.fail_json(msg="Schema '%s' does not exist" % schema)