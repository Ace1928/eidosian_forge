from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class BackendServiceIap(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({'enabled': self.request.get('enabled'), 'oauth2ClientId': self.request.get('oauth2_client_id'), 'oauth2ClientSecret': self.request.get('oauth2_client_secret')})

    def from_response(self):
        return remove_nones_from_dict({'enabled': self.request.get('enabled'), 'oauth2ClientId': self.request.get('oauth2ClientId'), 'oauth2ClientSecret': self.request.get('oauth2ClientSecret')})