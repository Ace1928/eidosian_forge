from __future__ import absolute_import, division, print_function
import json
import os
import re
import traceback
from ansible.module_utils.common.text.converters import to_text, to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import Request
class RestOperation(object):

    def __init__(self, session, uri, method, parameters=None):
        self.session = session
        self.method = method
        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = parameters
        self.url = '{scheme}://{host}{base_path}{uri}'.format(scheme='https', host=session._spec.get('host'), base_path=session._spec.get('basePath'), uri=uri)

    def restmethod(self, *args, **kwargs):
        """Do the hard work of making the request here"""
        if self.parameters:
            path_parameters = {}
            body_parameters = {}
            query_parameters = {}
            for x in self.parameters:
                expected_location = x.get('in')
                key_name = x.get('name', None)
                key_value = kwargs.get(key_name, None)
                if expected_location == 'path' and key_name and key_value:
                    path_parameters.update({key_name: key_value})
                elif expected_location == 'body' and key_name and key_value:
                    body_parameters.update({key_name: key_value})
                elif expected_location == 'query' and key_name and key_value:
                    query_parameters.update({key_name: key_value})
            if len(body_parameters.keys()) >= 1:
                body_parameters = body_parameters.get(list(body_parameters.keys())[0])
            else:
                body_parameters = None
        else:
            path_parameters = {}
            query_parameters = {}
            body_parameters = None
        url = self.url.format(**path_parameters)
        if query_parameters:
            url = url + '?' + urlencode(query_parameters)
        try:
            if body_parameters:
                body_parameters_json = json.dumps(body_parameters)
                response = self.session.request.open(method=self.method, url=url, data=body_parameters_json)
            else:
                response = self.session.request.open(method=self.method, url=url)
            request_error = False
        except HTTPError as e:
            response = e
            request_error = True
        try:
            result_code = response.getcode()
            result = json.loads(response.read())
        except ValueError:
            result = {}
        if result or result == {}:
            if result_code and result_code < 400:
                return result
            else:
                raise RestOperationException(result)
        raise RestOperationException({'status': result_code, 'errors': [{'message': 'REST Operation Failed'}]})