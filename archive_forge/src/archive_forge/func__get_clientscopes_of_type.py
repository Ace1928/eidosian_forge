from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def _get_clientscopes_of_type(self, realm, url_template, scope_type, client_id=None):
    """Fetch the name and ID of all clientscopes on the Keycloak server.

        To fetch the full data of the client scope, make a subsequent call to
        get_clientscope_by_clientscopeid, passing in the ID of the client scope you wish to return.

        :param realm: Realm in which the clientscope resides.
        :param url_template the template for the right type
        :param scope_type this can be either optional or default
        :param client_id: The client in which the clientscope resides.
        :return The clientscopes of the specified type of this realm
        """
    if client_id is None:
        clientscopes_url = url_template.format(url=self.baseurl, realm=realm)
        try:
            return json.loads(to_native(open_url(clientscopes_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
        except Exception as e:
            self.fail_open_url(e, msg='Could not fetch list of %s clientscopes in realm %s: %s' % (scope_type, realm, str(e)))
    else:
        cid = self.get_client_id(client_id=client_id, realm=realm)
        clientscopes_url = url_template.format(url=self.baseurl, realm=realm, cid=cid)
        try:
            return json.loads(to_native(open_url(clientscopes_url, method='GET', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, validate_certs=self.validate_certs).read()))
        except Exception as e:
            self.fail_open_url(e, msg='Could not fetch list of %s clientscopes in client %s: %s' % (scope_type, client_id, clientscopes_url))