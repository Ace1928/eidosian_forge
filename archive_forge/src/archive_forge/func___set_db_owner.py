from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def __set_db_owner(self):
    """Set the database owner."""
    query = 'ALTER DATABASE "%s" OWNER TO "%s"' % (self.obj_name, self.role)
    self.changed = exec_sql(self, query, return_bool=True)