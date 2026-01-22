from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class RegionBackendServiceOutlierdetection(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'baseEjectionTime': RegionBackendServiceBaseejectiontime(self.request.get('base_ejection_time', {}), self.module).to_request(), u'consecutiveErrors': self.request.get('consecutive_errors'), u'consecutiveGatewayFailure': self.request.get('consecutive_gateway_failure'), u'enforcingConsecutiveErrors': self.request.get('enforcing_consecutive_errors'), u'enforcingConsecutiveGatewayFailure': self.request.get('enforcing_consecutive_gateway_failure'), u'enforcingSuccessRate': self.request.get('enforcing_success_rate'), u'interval': RegionBackendServiceInterval(self.request.get('interval', {}), self.module).to_request(), u'maxEjectionPercent': self.request.get('max_ejection_percent'), u'successRateMinimumHosts': self.request.get('success_rate_minimum_hosts'), u'successRateRequestVolume': self.request.get('success_rate_request_volume'), u'successRateStdevFactor': self.request.get('success_rate_stdev_factor')})

    def from_response(self):
        return remove_nones_from_dict({u'baseEjectionTime': RegionBackendServiceBaseejectiontime(self.request.get(u'baseEjectionTime', {}), self.module).from_response(), u'consecutiveErrors': self.request.get(u'consecutiveErrors'), u'consecutiveGatewayFailure': self.request.get(u'consecutiveGatewayFailure'), u'enforcingConsecutiveErrors': self.request.get(u'enforcingConsecutiveErrors'), u'enforcingConsecutiveGatewayFailure': self.request.get(u'enforcingConsecutiveGatewayFailure'), u'enforcingSuccessRate': self.request.get(u'enforcingSuccessRate'), u'interval': RegionBackendServiceInterval(self.request.get(u'interval', {}), self.module).from_response(), u'maxEjectionPercent': self.request.get(u'maxEjectionPercent'), u'successRateMinimumHosts': self.request.get(u'successRateMinimumHosts'), u'successRateRequestVolume': self.request.get(u'successRateRequestVolume'), u'successRateStdevFactor': self.request.get(u'successRateStdevFactor')})