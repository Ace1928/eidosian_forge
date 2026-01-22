from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
class CloudFunctionEventtrigger(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({'eventType': self.request.get('event_type'), 'resource': self.request.get('resource'), 'service': self.request.get('service')})

    def from_response(self):
        return remove_nones_from_dict({'eventType': self.request.get('eventType'), 'resource': self.request.get('resource'), 'service': self.request.get('service')})