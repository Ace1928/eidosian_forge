from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class RegionBackendServiceConsistenthash(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'httpCookie': RegionBackendServiceHttpcookie(self.request.get('http_cookie', {}), self.module).to_request(), u'httpHeaderName': self.request.get('http_header_name'), u'minimumRingSize': self.request.get('minimum_ring_size')})

    def from_response(self):
        return remove_nones_from_dict({u'httpCookie': RegionBackendServiceHttpcookie(self.request.get(u'httpCookie', {}), self.module).from_response(), u'httpHeaderName': self.request.get(u'httpHeaderName'), u'minimumRingSize': self.request.get(u'minimumRingSize')})