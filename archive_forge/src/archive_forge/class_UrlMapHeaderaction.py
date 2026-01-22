from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class UrlMapHeaderaction(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'requestHeadersToRemove': self.request.get('request_headers_to_remove'), u'requestHeadersToAdd': UrlMapRequestheaderstoaddArray(self.request.get('request_headers_to_add', []), self.module).to_request(), u'responseHeadersToRemove': self.request.get('response_headers_to_remove'), u'responseHeadersToAdd': UrlMapResponseheaderstoaddArray(self.request.get('response_headers_to_add', []), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'requestHeadersToRemove': self.request.get(u'requestHeadersToRemove'), u'requestHeadersToAdd': UrlMapRequestheaderstoaddArray(self.request.get(u'requestHeadersToAdd', []), self.module).from_response(), u'responseHeadersToRemove': self.request.get(u'responseHeadersToRemove'), u'responseHeadersToAdd': UrlMapResponseheaderstoaddArray(self.request.get(u'responseHeadersToAdd', []), self.module).from_response()})