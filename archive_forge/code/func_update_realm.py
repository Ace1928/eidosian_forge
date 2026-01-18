from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def update_realm(self, realmrep, realm='master'):
    """ Update an existing realm
        :param realmrep: corresponding (partial/full) realm representation with updates
        :param realm: realm to be updated in Keycloak
        :return: HTTPResponse object on success
        """
    realm_url = URL_REALM.format(url=self.baseurl, realm=realm)
    try:
        return open_url(realm_url, method='PUT', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, data=json.dumps(realmrep), validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Could not update realm %s: %s' % (realm, str(e)), exception=traceback.format_exc())