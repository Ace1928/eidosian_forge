from __future__ import absolute_import, division, print_function
import json
import re
import traceback
from ansible.module_utils.six import PY3
from ansible.module_utils._text import to_text
from ansible.module_utils.connection import ConnectionError
from ansible.plugins.httpapi import HttpApiBase
from copy import copy
def _get_login_domain_id(self, domain_name):
    """Get a domain and return its id"""
    if domain_name is None:
        return None
    method = 'GET'
    path = '/mso/api/v1/auth/login-domains'
    full_path = self.connection.get_option('host') + path
    response, data = self.connection.send(path, None, method=method, headers=self.headers)
    if data is not None:
        response_data = self._response_to_json(data)
        domains = response_data.get('domains')
        if domains is not None:
            for domain in domains:
                if domain.get('name') == domain_name:
                    if 'id' in domain:
                        return domain.get('id')
                    else:
                        self.error = dict(code=-1, message="Login domain lookup failed for domain '{0}': {1}".format(domain_name, domain))
                        raise ConnectionError(json.dumps(self._verify_response(None, method, full_path, None)))
            self.error = dict(code=-1, message="Login domain '{0}' is not a valid domain name.".format(domain_name))
            raise ConnectionError(json.dumps(self._verify_response(None, method, full_path, None)))
        else:
            self.error = dict(code=-1, message="Key 'domains' missing from data")
            raise ConnectionError(json.dumps(self._verify_response(None, method, full_path, None)))