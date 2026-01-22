from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class TriggerPubsubconfig(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'topic': self.request.get('topic'), u'service_account_email': self.request.get('service_account_email')})

    def from_response(self):
        return remove_nones_from_dict({u'topic': self.request.get(u'topic'), u'service_account_email': self.request.get(u'service_account_email')})