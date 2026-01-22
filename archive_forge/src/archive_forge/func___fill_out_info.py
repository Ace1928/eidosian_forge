from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
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