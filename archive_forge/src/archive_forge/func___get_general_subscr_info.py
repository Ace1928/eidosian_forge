from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def __get_general_subscr_info(self):
    """Get and return general subscription information.

        Returns:
            Dict with subscription information if successful, False otherwise.
        """
    query = "SELECT obj_description(s.oid, 'pg_subscription') AS comment, d.datname, r.rolname, s.subenabled, s.subconninfo, s.subslotname, s.subsynccommit, s.subpublications FROM pg_catalog.pg_subscription s JOIN pg_catalog.pg_database d ON s.subdbid = d.oid JOIN pg_catalog.pg_roles AS r ON s.subowner = r.oid WHERE s.subname = %(name)s AND d.datname = %(db)s"
    result = exec_sql(self, query, query_params={'name': self.name, 'db': self.db}, add_to_executed=False)
    if result:
        return result[0]
    else:
        return False