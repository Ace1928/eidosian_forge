from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class MetricBucketoptions(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'linearBuckets': MetricLinearbuckets(self.request.get('linear_buckets', {}), self.module).to_request(), u'exponentialBuckets': MetricExponentialbuckets(self.request.get('exponential_buckets', {}), self.module).to_request(), u'explicitBuckets': MetricExplicitbuckets(self.request.get('explicit_buckets', {}), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'linearBuckets': MetricLinearbuckets(self.request.get(u'linearBuckets', {}), self.module).from_response(), u'exponentialBuckets': MetricExponentialbuckets(self.request.get(u'exponentialBuckets', {}), self.module).from_response(), u'explicitBuckets': MetricExplicitbuckets(self.request.get(u'explicitBuckets', {}), self.module).from_response()})