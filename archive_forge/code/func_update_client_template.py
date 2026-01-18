from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def update_client_template(self, id, clienttrep, realm='master'):
    """ Update an existing client template
        :param id: id (not name) of client template to be updated in Keycloak
        :param clienttrep: corresponding (partial/full) client template representation with updates
        :param realm: realm the client template is in
        :return: HTTPResponse object on success
        """
    url = URL_CLIENTTEMPLATE.format(url=self.baseurl, realm=realm, id=id)
    try:
        return open_url(url, method='PUT', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, data=json.dumps(clienttrep), validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Could not update client template %s in realm %s: %s' % (id, realm, str(e)))