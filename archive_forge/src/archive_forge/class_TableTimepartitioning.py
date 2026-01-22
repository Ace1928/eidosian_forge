from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class TableTimepartitioning(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'expirationMs': self.request.get('expiration_ms'), u'field': self.request.get('field'), u'type': self.request.get('type')})

    def from_response(self):
        return remove_nones_from_dict({u'expirationMs': self.request.get(u'expirationMs'), u'field': self.request.get(u'field'), u'type': self.request.get(u'type')})