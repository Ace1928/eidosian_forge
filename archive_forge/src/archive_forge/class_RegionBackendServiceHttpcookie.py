from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class RegionBackendServiceHttpcookie(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'ttl': RegionBackendServiceTtl(self.request.get('ttl', {}), self.module).to_request(), u'name': self.request.get('name'), u'path': self.request.get('path')})

    def from_response(self):
        return remove_nones_from_dict({u'ttl': RegionBackendServiceTtl(self.request.get(u'ttl', {}), self.module).from_response(), u'name': self.request.get(u'name'), u'path': self.request.get(u'path')})