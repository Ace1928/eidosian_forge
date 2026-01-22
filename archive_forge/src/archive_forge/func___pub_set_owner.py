from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
def __pub_set_owner(self, role, check_mode=False):
    """Set a publication owner.

        Args:
            role (str): Role (user) name that needs to be set as a publication owner.

        Kwargs:
            check_mode (bool): If True, don't actually change anything,
                just make SQL, add it to ``self.executed_queries`` and return True.

        Returns:
            True if successful, False otherwise.
        """
    query = 'ALTER PUBLICATION %s OWNER TO "%s"' % (pg_quote_identifier(self.name, 'publication'), role)
    return self.__exec_sql(query, check_mode=check_mode)