from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.mysql.plugins.module_utils.mysql import (
from ansible_collections.community.mysql.plugins.module_utils.user import (
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
def __remove_member(self, user, check_mode=False):
    """Remove a member from a role.

        Args:
            user (str): Role member to remove.
            check_mode (bool): If True, just returns True and does nothing.

        Returns:
            bool: True if the state has changed, False if has not.
        """
    if check_mode:
        return True
    self.cursor.execute(*self.q_builder.role_revoke(user))
    return True