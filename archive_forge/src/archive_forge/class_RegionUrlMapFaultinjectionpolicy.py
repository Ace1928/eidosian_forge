from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class RegionUrlMapFaultinjectionpolicy(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'abort': RegionUrlMapAbort(self.request.get('abort', {}), self.module).to_request(), u'delay': RegionUrlMapDelay(self.request.get('delay', {}), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'abort': RegionUrlMapAbort(self.request.get(u'abort', {}), self.module).from_response(), u'delay': RegionUrlMapDelay(self.request.get(u'delay', {}), self.module).from_response()})