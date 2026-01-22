from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class MetricExponentialbuckets(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'numFiniteBuckets': self.request.get('num_finite_buckets'), u'growthFactor': self.request.get('growth_factor'), u'scale': self.request.get('scale')})

    def from_response(self):
        return remove_nones_from_dict({u'numFiniteBuckets': self.request.get(u'numFiniteBuckets'), u'growthFactor': self.request.get(u'growthFactor'), u'scale': self.request.get(u'scale')})