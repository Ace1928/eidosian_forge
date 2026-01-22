from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
class MetricMetricdescriptor(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'unit': self.request.get('unit'), u'valueType': self.request.get('value_type'), u'metricKind': self.request.get('metric_kind'), u'labels': MetricLabelsArray(self.request.get('labels', []), self.module).to_request(), u'displayName': self.request.get('display_name')})

    def from_response(self):
        return remove_nones_from_dict({u'unit': self.request.get(u'unit'), u'valueType': self.request.get(u'valueType'), u'metricKind': self.request.get(u'metricKind'), u'labels': MetricLabelsArray(self.request.get(u'labels', []), self.module).from_response(), u'displayName': self.request.get(u'displayName')})