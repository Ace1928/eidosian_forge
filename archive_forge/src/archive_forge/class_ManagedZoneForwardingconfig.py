from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class ManagedZoneForwardingconfig(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'targetNameServers': ManagedZoneTargetnameserversArray(self.request.get('target_name_servers', []), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'targetNameServers': ManagedZoneTargetnameserversArray(self.request.get(u'targetNameServers', []), self.module).from_response()})