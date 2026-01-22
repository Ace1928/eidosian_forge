from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class ClusterLegacyabac(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'enabled': self.request.get('enabled')})

    def from_response(self):
        return remove_nones_from_dict({u'enabled': self.request.get(u'enabled')})