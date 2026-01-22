from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
class DbServer:
    """Class to fetch information from a database.

    Args:
        module (AnsibleModule): Object of the AnsibleModule class.
        cursor (cursor): Cursor object of a database Python connector.

    Attributes:
        module (AnsibleModule): Object of the AnsibleModule class.
        cursor (cursor): Cursor object of a database Python connector.
        role_impl (library): Corresponding library depending
            on a server type (MariaDB or MySQL)
        mariadb (bool): True if MariaDB, False otherwise.
        roles_supported (bool): True if roles are supported, False otherwise.
        users (set): Set of users existing in a DB in the form (username, hostname).
    """

    def __init__(self, module, cursor):
        self.module = module
        self.cursor = cursor
        self.role_impl = self.get_implementation()
        self.mariadb = self.role_impl.is_mariadb()
        self.roles_supported = self.role_impl.supports_roles(self.cursor)
        self.users = set(self.__get_users())

    def is_mariadb(self):
        """Get info whether a DB server is a MariaDB instance.

        Returns:
            self.mariadb: Attribute value.
        """
        return self.mariadb

    def supports_roles(self):
        """Get info whether a DB server supports roles.

        Returns:
            self.roles_supported: Attribute value.
        """
        return self.roles_supported

    def get_implementation(self):
        """Get a current server implementation depending on its type.

        Returns:
            library: Depending on a server type (MySQL or MariaDB).
        """
        self.cursor.execute('SELECT VERSION()')
        if 'mariadb' in self.cursor.fetchone()[0].lower():
            import ansible_collections.community.mysql.plugins.module_utils.implementations.mariadb.role as role_impl
        else:
            import ansible_collections.community.mysql.plugins.module_utils.implementations.mysql.role as role_impl
        return role_impl

    def check_users_in_db(self, users):
        """Check if users exist in a database.

        Args:
            users (list): List of tuples (username, hostname) to check.
        """
        for user in users:
            if user not in self.users:
                msg = 'User / role `%s` with host `%s` does not exist' % (user[0], user[1])
                self.module.fail_json(msg=msg)

    def filter_existing_users(self, users):
        for user in users:
            if user in self.users:
                yield user

    def __get_users(self):
        """Get users.

        Returns:
            list: List of tuples (username, hostname).
        """
        self.cursor.execute('SELECT User, Host FROM mysql.user')
        return self.cursor.fetchall()

    def get_users(self):
        """Get set of tuples (username, hostname) existing in a DB.

        Returns:
            self.users: Attribute value.
        """
        return self.users

    def get_grants(self, user, host):
        """Get grants.

        Args:
            user (str): User name
            host (str): Host name

        Returns:
            list: List of tuples like [(grant1,), (grant2,), ... ].
        """
        if host:
            self.cursor.execute('SHOW GRANTS FOR %s@%s', (user, host))
        else:
            self.cursor.execute('SHOW GRANTS FOR %s', (user,))
        return self.cursor.fetchall()