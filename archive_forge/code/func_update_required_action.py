from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def update_required_action(self, alias, rep, realm='master'):
    """
        Update required action.
        :param alias: Alias of required action.
        :param rep:   JSON describing new state of required action.
        :param realm: Realm name (not id).
        :return:      HTTPResponse object on success.
        """
    try:
        return open_url(URL_AUTHENTICATION_REQUIRED_ACTIONS_ALIAS.format(url=self.baseurl, alias=quote(alias, safe=''), realm=realm), method='PUT', http_agent=self.http_agent, headers=self.restheaders, data=json.dumps(rep), timeout=self.connection_timeout, validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Unable to update required action %s in realm %s: %s' % (alias, realm, str(e)))