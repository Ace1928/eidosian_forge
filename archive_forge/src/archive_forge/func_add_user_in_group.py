from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def add_user_in_group(self, user_id, group_id, realm='master'):
    """
        Add a user to a group.
        :param user_id: User ID
        :param group_id: Group Id to add the user to.
        :param realm: Realm
        :return: HTTP Response
        """
    try:
        user_group_url = URL_USER_GROUP.format(url=self.baseurl, realm=realm, id=user_id, group_id=group_id)
        return open_url(user_group_url, method='PUT', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Could not add user %s in group %s in realm %s: %s' % (user_id, group_id, realm, str(e)))