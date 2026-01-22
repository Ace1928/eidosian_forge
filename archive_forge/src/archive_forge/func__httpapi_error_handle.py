from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six import iteritems
from ansible.module_utils.six.moves.urllib.parse import urlencode
def _httpapi_error_handle(self, method, uri, payload=None):
    try:
        code, response = self.connection.send_request(method, uri, payload=payload)
        if code == 404:
            if to_text('Object not found') in to_text(response) or to_text('Could not find object') in to_text(response):
                return {}
        if not (code >= 200 and code < 300):
            self.module.fail_json(msg='Splunk httpapi returned error {0} with message {1}'.format(code, response))
        return response
    except ConnectionError as e:
        self.module.fail_json(msg='connection error occurred: {0}'.format(e))
    except CertificateError as e:
        self.module.fail_json(msg='certificate error occurred: {0}'.format(e))
    except ValueError as e:
        try:
            self.module.fail_json(msg='certificate not found: {0}'.format(e))
        except AttributeError:
            pass