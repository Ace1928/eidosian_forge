from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class ManagedZoneDnssecconfig(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'kind': self.request.get('kind'), u'nonExistence': self.request.get('non_existence'), u'state': self.request.get('state'), u'defaultKeySpecs': ManagedZoneDefaultkeyspecsArray(self.request.get('default_key_specs', []), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'kind': self.request.get(u'kind'), u'nonExistence': self.request.get(u'nonExistence'), u'state': self.request.get(u'state'), u'defaultKeySpecs': ManagedZoneDefaultkeyspecsArray(self.request.get(u'defaultKeySpecs', []), self.module).from_response()})