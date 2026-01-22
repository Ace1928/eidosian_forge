from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def __pub_add_table(self, table, check_mode=False):
    """Add a table to the publication.

        Args:
            table (str): Table name.

        Kwargs:
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            True if successful, False otherwise.
        """
    query = 'ALTER PUBLICATION %s ADD TABLE %s' % (pg_quote_identifier(self.name, 'publication'), pg_quote_identifier(table, 'table'))
    return self.__exec_sql(query, check_mode=check_mode)