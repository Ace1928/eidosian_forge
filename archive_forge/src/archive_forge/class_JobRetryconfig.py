from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class JobRetryconfig(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'retryCount': self.request.get('retry_count'), u'maxRetryDuration': self.request.get('max_retry_duration'), u'minBackoffDuration': self.request.get('min_backoff_duration'), u'maxBackoffDuration': self.request.get('max_backoff_duration'), u'maxDoublings': self.request.get('max_doublings')})

    def from_response(self):
        return remove_nones_from_dict({u'retryCount': self.request.get(u'retryCount'), u'maxRetryDuration': self.request.get(u'maxRetryDuration'), u'minBackoffDuration': self.request.get(u'minBackoffDuration'), u'maxBackoffDuration': self.request.get(u'maxBackoffDuration'), u'maxDoublings': self.request.get(u'maxDoublings')})