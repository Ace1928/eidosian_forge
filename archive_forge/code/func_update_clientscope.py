from __future__ import absolute_import, division, print_function
import json
import traceback
import copy
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import urlencode, quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.common.text.converters import to_native, to_text
def update_clientscope(self, clientscoperep, realm='master'):
    """ Update an existing clientscope.

        :param grouprep: A GroupRepresentation of the updated group.
        :return HTTPResponse object on success
        """
    clientscope_url = URL_CLIENTSCOPE.format(url=self.baseurl, realm=realm, id=clientscoperep['id'])
    try:
        return open_url(clientscope_url, method='PUT', http_agent=self.http_agent, headers=self.restheaders, timeout=self.connection_timeout, data=json.dumps(clientscoperep), validate_certs=self.validate_certs)
    except Exception as e:
        self.fail_open_url(e, msg='Could not update clientscope %s in realm %s: %s' % (clientscoperep['name'], realm, str(e)))