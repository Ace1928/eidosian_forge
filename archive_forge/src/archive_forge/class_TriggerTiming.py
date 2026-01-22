from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class TriggerTiming(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'startTime': self.request.get('start_time'), u'endTime': self.request.get('end_time')})

    def from_response(self):
        return remove_nones_from_dict({u'startTime': self.request.get(u'startTime'), u'endTime': self.request.get(u'endTime')})