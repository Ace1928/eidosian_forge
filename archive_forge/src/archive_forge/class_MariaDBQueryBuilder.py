from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
class MariaDBQueryBuilder:
    """Class to build and return queries specific to MariaDB.

    Args:
        name (str): Role name.

    Attributes:
        name (str): Role name.
    """

    def __init__(self, name):
        self.name = name

    def role_exists(self):
        """Return a query to check if a role with self.name exists in a database.

        Returns:
            tuple: (query_string, tuple_containing_parameters).
        """
        return ("SELECT count(*) FROM mysql.user WHERE user = %s AND is_role  = 'Y'", (self.name,))

    def role_grant(self, user):
        """Return a query to grant a role to a user or role.

        Args:
            user (tuple): User / role to grant the role to in the form (username, hostname).

        Returns:
            tuple: (query_string, tuple_containing_parameters).
        """
        if user[1]:
            return ('GRANT %s TO %s@%s', (self.name, user[0], user[1]))
        else:
            return ('GRANT %s TO %s', (self.name, user[0]))

    def role_revoke(self, user):
        """Return a query to revoke a role from a user or role.

        Args:
            user (tuple): User / role to revoke the role from in the form (username, hostname).

        Returns:
            tuple: (query_string, tuple_containing_parameters).
        """
        if user[1]:
            return ('REVOKE %s FROM %s@%s', (self.name, user[0], user[1]))
        else:
            return ('REVOKE %s FROM %s', (self.name, user[0]))

    def role_create(self, admin=None):
        """Return a query to create a role.

        Args:
            admin (tuple): Admin user in the form (username, hostname).

        Returns:
            tuple: (query_string, tuple_containing_parameters).
        """
        if not admin:
            return ('CREATE ROLE %s', (self.name,))
        if admin[1]:
            return ('CREATE ROLE %s WITH ADMIN %s@%s', (self.name, admin[0], admin[1]))
        else:
            return ('CREATE ROLE %s WITH ADMIN %s', (self.name, admin[0]))