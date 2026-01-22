from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleConnectionFailure
from ansible.module_utils.basic import to_text
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible_collections.ansible.netcommon.plugins.plugin_utils.httpapi_base import HttpApiBase
class HttpApi(HttpApiBase):

    def send_request(self, request_method, path, payload=None):
        try:
            self._display_request(request_method, path)
            response, response_data = self.connection.send(path, payload, method=request_method, headers=BASE_HEADERS, force_basic_auth=True)
            value = self._get_response_value(response_data)
            return (response.getcode(), self._response_to_json(value))
        except AnsibleConnectionFailure as e:
            self.connection.queue_message('vvv', 'AnsibleConnectionFailure: %s' % e)
            if to_text('Could not connect to') in to_text(e):
                raise
            if to_text('401') in to_text(e):
                return (401, 'Authentication failure')
            else:
                return (404, 'Object not found')
        except HTTPError as e:
            error = json.loads(e.read())
            return (e.code, error)

    def _display_request(self, request_method, path):
        self.connection.queue_message('vvvv', 'Web Services: %s %s/%s' % (request_method, self.connection._url, path))

    def _get_response_value(self, response_data):
        return to_text(response_data.getvalue())

    def _response_to_json(self, response_text):
        try:
            return json.loads(response_text) if response_text else {}
        except ValueError:
            raise ConnectionError('Invalid JSON response: %s' % response_text)