from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
class MySQLRoleImpl:
    """Class to work with MySQL role implementation.

    Args:
        module (AnsibleModule): Object of the AnsibleModule class.
        cursor (cursor): Cursor object of a database Python connector.
        name (str): Role name.
        host (str): Role host.

    Attributes:
        module (AnsibleModule): Object of the AnsibleModule class.
        cursor (cursor): Cursor object of a database Python connector.
        name (str): Role name.
        host (str): Role host.
    """

    def __init__(self, module, cursor, name, host):
        self.module = module
        self.cursor = cursor
        self.name = name
        self.host = host

    def set_default_role_all(self, user):
        """Run 'SET DEFAULT ROLE ALL TO' a user.

        Args:
            user (tuple): User / role to run the command against in the form (username, hostname).
        """
        if user[1]:
            self.cursor.execute('SET DEFAULT ROLE ALL TO %s@%s', (user[0], user[1]))
        else:
            self.cursor.execute('SET DEFAULT ROLE ALL TO %s', (user[0],))

    def get_admin(self):
        """Get a current admin of a role.

        Not supported by MySQL, so ignored here.
        """
        pass

    def set_admin(self, admin):
        """Set an admin of a role.

        Not supported by MySQL, so ignored here.

        TODO: Implement the feature if this gets supported.

        Args:
            admin (tuple): Admin user of the role in the form (username, hostname).
        """
        pass