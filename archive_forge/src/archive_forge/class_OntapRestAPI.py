from __future__ import absolute_import, division, print_function
import json
import os
import random
import mimetypes
from pprint import pformat
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.urls import open_url
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils._text import to_native
import ssl
class OntapRestAPI(object):

    def __init__(self, module, timeout=60):
        self.module = module
        self.username = self.module.params['username']
        self.password = self.module.params['password']
        self.hostname = self.module.params['hostname']
        self.use_rest = self.module.params['use_rest']
        self.verify = self.module.params['validate_certs']
        self.timeout = timeout
        self.url = 'https://' + self.hostname + '/api/'
        self.errors = list()
        self.debug_logs = list()
        self.check_required_library()

    def check_required_library(self):
        if not HAS_REQUESTS:
            self.module.fail_json(msg=missing_required_lib('requests'))

    def send_request(self, method, api, params, json=None, return_status_code=False):
        """ send http request and process reponse, including error conditions """
        url = self.url + api
        status_code = None
        content = None
        json_dict = None
        json_error = None
        error_details = None

        def get_json(response):
            """ extract json, and error message if present """
            try:
                json = response.json()
            except ValueError:
                return (None, None)
            error = json.get('error')
            return (json, error)
        try:
            response = requests.request(method, url, verify=self.verify, auth=(self.username, self.password), params=params, timeout=self.timeout, json=json)
            content = response.content
            status_code = response.status_code
            response.raise_for_status()
            json_dict, json_error = get_json(response)
        except requests.exceptions.HTTPError as err:
            __, json_error = get_json(response)
            if json_error is None:
                self.log_error(status_code, 'HTTP error: %s' % err)
                error_details = str(err)
        except requests.exceptions.ConnectionError as err:
            self.log_error(status_code, 'Connection error: %s' % err)
            error_details = str(err)
        except Exception as err:
            self.log_error(status_code, 'Other error: %s' % err)
            error_details = str(err)
        if json_error is not None:
            self.log_error(status_code, 'Endpoint error: %d: %s' % (status_code, json_error))
            error_details = json_error
        self.log_debug(status_code, content)
        if return_status_code:
            return (status_code, error_details)
        return (json_dict, error_details)

    def get(self, api, params):
        method = 'GET'
        return self.send_request(method, api, params)

    def post(self, api, data, params=None):
        method = 'POST'
        return self.send_request(method, api, params, json=data)

    def patch(self, api, data, params=None):
        method = 'PATCH'
        return self.send_request(method, api, params, json=data)

    def delete(self, api, data, params=None):
        method = 'DELETE'
        return self.send_request(method, api, params, json=data)

    def _is_rest(self, used_unsupported_rest_properties=None):
        if self.use_rest == 'Always':
            if used_unsupported_rest_properties:
                error = "REST API currently does not support '%s'" % ', '.join(used_unsupported_rest_properties)
                return (True, error)
            else:
                return (True, None)
        if self.use_rest == 'Never' or used_unsupported_rest_properties:
            return (False, None)
        method = 'HEAD'
        api = 'cluster/software'
        status_code, __ = self.send_request(method, api, params=None, return_status_code=True)
        if status_code == 200:
            return (True, None)
        return (False, None)

    def is_rest(self, used_unsupported_rest_properties=None):
        """ only return error if there is a reason to """
        use_rest, error = self._is_rest(used_unsupported_rest_properties)
        if used_unsupported_rest_properties is None:
            return use_rest
        return (use_rest, error)

    def log_error(self, status_code, message):
        self.errors.append(message)
        self.debug_logs.append((status_code, message))

    def log_debug(self, status_code, content):
        self.debug_logs.append((status_code, content))