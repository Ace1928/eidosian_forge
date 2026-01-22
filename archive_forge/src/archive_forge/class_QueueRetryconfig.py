from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
class QueueRetryconfig(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'maxAttempts': self.request.get('max_attempts'), u'maxRetryDuration': self.request.get('max_retry_duration'), u'minBackoff': self.request.get('min_backoff'), u'maxBackoff': self.request.get('max_backoff'), u'maxDoublings': self.request.get('max_doublings')})

    def from_response(self):
        return remove_nones_from_dict({u'maxAttempts': self.request.get(u'maxAttempts'), u'maxRetryDuration': self.request.get(u'maxRetryDuration'), u'minBackoff': self.request.get(u'minBackoff'), u'maxBackoff': self.request.get(u'maxBackoff'), u'maxDoublings': self.request.get(u'maxDoublings')})