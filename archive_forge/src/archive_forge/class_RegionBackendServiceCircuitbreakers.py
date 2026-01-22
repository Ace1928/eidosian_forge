from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class RegionBackendServiceCircuitbreakers(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'maxRequestsPerConnection': self.request.get('max_requests_per_connection'), u'maxConnections': self.request.get('max_connections'), u'maxPendingRequests': self.request.get('max_pending_requests'), u'maxRequests': self.request.get('max_requests'), u'maxRetries': self.request.get('max_retries')})

    def from_response(self):
        return remove_nones_from_dict({u'maxRequestsPerConnection': self.request.get(u'maxRequestsPerConnection'), u'maxConnections': self.request.get(u'maxConnections'), u'maxPendingRequests': self.request.get(u'maxPendingRequests'), u'maxRequests': self.request.get(u'maxRequests'), u'maxRetries': self.request.get(u'maxRetries')})