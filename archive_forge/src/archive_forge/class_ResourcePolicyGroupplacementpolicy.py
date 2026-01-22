from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class ResourcePolicyGroupplacementpolicy(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'vmCount': self.request.get('vm_count'), u'availabilityDomainCount': self.request.get('availability_domain_count'), u'collocation': self.request.get('collocation')})

    def from_response(self):
        return remove_nones_from_dict({u'vmCount': self.request.get(u'vmCount'), u'availabilityDomainCount': self.request.get(u'availabilityDomainCount'), u'collocation': self.request.get(u'collocation')})