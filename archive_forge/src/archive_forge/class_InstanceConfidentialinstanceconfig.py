from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
class InstanceConfidentialinstanceconfig(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'enableConfidentialCompute': self.request.get('enable_confidential_compute')})

    def from_response(self):
        return remove_nones_from_dict({u'enableConfidentialCompute': self.request.get(u'enableConfidentialCompute')})