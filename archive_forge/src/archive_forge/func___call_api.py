from __future__ import absolute_import
import os
import re
import json
import mimetypes
import tempfile
from multiprocessing.pool import ThreadPool
from datetime import date, datetime
from six import PY3, integer_types, iteritems, text_type
from six.moves.urllib.parse import quote
from . import models
from .configuration import Configuration
from .rest import ApiException, RESTClientObject
def __call_api(self, resource_path, method, path_params=None, query_params=None, header_params=None, body=None, post_params=None, files=None, response_type=None, auth_settings=None, _return_http_data_only=None, collection_formats=None, _preload_content=True, _request_timeout=None):
    config = self.configuration
    header_params = header_params or {}
    header_params.update(self.default_headers)
    if self.cookie:
        header_params['Cookie'] = self.cookie
    if header_params:
        header_params = self.sanitize_for_serialization(header_params)
        header_params = dict(self.parameters_to_tuples(header_params, collection_formats))
    if path_params:
        path_params = self.sanitize_for_serialization(path_params)
        path_params = self.parameters_to_tuples(path_params, collection_formats)
        for k, v in path_params:
            resource_path = resource_path.replace('{%s}' % k, quote(str(v), safe=config.safe_chars_for_path_param))
    if query_params:
        query_params = self.sanitize_for_serialization(query_params)
        query_params = self.parameters_to_tuples(query_params, collection_formats)
    if post_params or files:
        post_params = self.prepare_post_parameters(post_params, files)
        post_params = self.sanitize_for_serialization(post_params)
        post_params = self.parameters_to_tuples(post_params, collection_formats)
    self.update_params_for_auth(header_params, query_params, auth_settings)
    if body:
        body = self.sanitize_for_serialization(body)
    url = self.configuration.host + resource_path
    response_data = self.request(method, url, query_params=query_params, headers=header_params, post_params=post_params, body=body, _preload_content=_preload_content, _request_timeout=_request_timeout)
    self.last_response = response_data
    return_data = response_data
    if _preload_content:
        if response_type:
            return_data = self.deserialize(response_data, response_type)
        else:
            return_data = None
    if _return_http_data_only:
        return return_data
    else:
        return (return_data, response_data.status, response_data.getheaders())