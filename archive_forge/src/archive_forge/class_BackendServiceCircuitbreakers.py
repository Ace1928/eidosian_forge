from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class BackendServiceCircuitbreakers(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({'maxRequestsPerConnection': self.request.get('max_requests_per_connection'), 'maxConnections': self.request.get('max_connections'), 'maxPendingRequests': self.request.get('max_pending_requests'), 'maxRequests': self.request.get('max_requests'), 'maxRetries': self.request.get('max_retries')})

    def from_response(self):
        return remove_nones_from_dict({'maxRequestsPerConnection': self.request.get('maxRequestsPerConnection'), 'maxConnections': self.request.get('maxConnections'), 'maxPendingRequests': self.request.get('maxPendingRequests'), 'maxRequests': self.request.get('maxRequests'), 'maxRetries': self.request.get('maxRetries')})