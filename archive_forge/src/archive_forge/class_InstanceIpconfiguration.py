from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class InstanceIpconfiguration(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'ipv4Enabled': self.request.get('ipv4_enabled'), u'privateNetwork': self.request.get('private_network'), u'authorizedNetworks': InstanceAuthorizednetworksArray(self.request.get('authorized_networks', []), self.module).to_request(), u'requireSsl': self.request.get('require_ssl')})

    def from_response(self):
        return remove_nones_from_dict({u'ipv4Enabled': self.request.get(u'ipv4Enabled'), u'privateNetwork': self.request.get(u'privateNetwork'), u'authorizedNetworks': InstanceAuthorizednetworksArray(self.request.get(u'authorizedNetworks', []), self.module).from_response(), u'requireSsl': self.request.get(u'requireSsl')})