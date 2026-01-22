from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class RegionBackendServiceCachekeypolicy(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'includeHost': self.request.get('include_host'), u'includeProtocol': self.request.get('include_protocol'), u'includeQueryString': self.request.get('include_query_string'), u'queryStringBlacklist': self.request.get('query_string_blacklist'), u'queryStringWhitelist': self.request.get('query_string_whitelist')})

    def from_response(self):
        return remove_nones_from_dict({u'includeHost': self.request.get(u'includeHost'), u'includeProtocol': self.request.get(u'includeProtocol'), u'includeQueryString': self.request.get(u'includeQueryString'), u'queryStringBlacklist': self.request.get(u'queryStringBlacklist'), u'queryStringWhitelist': self.request.get(u'queryStringWhitelist')})