from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def update_user_groups_membership(self, userrep, groups, realm='master'):
    """
        Update user's group membership
        :param userrep: Representation of the user. This representation must include the ID.
        :param realm: Realm
        :return: True if group membership has been changed. False Otherwise.
        """
    changed = False
    try:
        user_existing_groups = self.get_user_groups(user_id=userrep['id'], realm=realm)
        groups_to_add_and_remove = self.extract_groups_to_add_to_and_remove_from_user(groups)
        if not is_struct_included(groups_to_add_and_remove['add'], user_existing_groups):
            realm_groups = self.get_groups(realm=realm)
            for realm_group in realm_groups:
                if 'name' in realm_group and realm_group['name'] in groups_to_add_and_remove['add']:
                    self.add_user_in_group(user_id=userrep['id'], group_id=realm_group['id'], realm=realm)
                    changed = True
                elif 'name' in realm_group and realm_group['name'] in groups_to_add_and_remove['remove']:
                    self.remove_user_from_group(user_id=userrep['id'], group_id=realm_group['id'], realm=realm)
                    changed = True
        return changed
    except Exception as e:
        self.module.fail_json(msg='Could not update group membership for user %s in realm %s: %s' % (userrep['id]'], realm, str(e)))