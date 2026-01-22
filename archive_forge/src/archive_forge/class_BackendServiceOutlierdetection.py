from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class BackendServiceOutlierdetection(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({'baseEjectionTime': BackendServiceBaseejectiontime(self.request.get('base_ejection_time', {}), self.module).to_request(), 'consecutiveErrors': self.request.get('consecutive_errors'), 'consecutiveGatewayFailure': self.request.get('consecutive_gateway_failure'), 'enforcingConsecutiveErrors': self.request.get('enforcing_consecutive_errors'), 'enforcingConsecutiveGatewayFailure': self.request.get('enforcing_consecutive_gateway_failure'), 'enforcingSuccessRate': self.request.get('enforcing_success_rate'), 'interval': BackendServiceInterval(self.request.get('interval', {}), self.module).to_request(), 'maxEjectionPercent': self.request.get('max_ejection_percent'), 'successRateMinimumHosts': self.request.get('success_rate_minimum_hosts'), 'successRateRequestVolume': self.request.get('success_rate_request_volume'), 'successRateStdevFactor': self.request.get('success_rate_stdev_factor')})

    def from_response(self):
        return remove_nones_from_dict({'baseEjectionTime': BackendServiceBaseejectiontime(self.request.get('baseEjectionTime', {}), self.module).from_response(), 'consecutiveErrors': self.request.get('consecutiveErrors'), 'consecutiveGatewayFailure': self.request.get('consecutiveGatewayFailure'), 'enforcingConsecutiveErrors': self.request.get('enforcingConsecutiveErrors'), 'enforcingConsecutiveGatewayFailure': self.request.get('enforcingConsecutiveGatewayFailure'), 'enforcingSuccessRate': self.request.get('enforcingSuccessRate'), 'interval': BackendServiceInterval(self.request.get('interval', {}), self.module).from_response(), 'maxEjectionPercent': self.request.get('maxEjectionPercent'), 'successRateMinimumHosts': self.request.get('successRateMinimumHosts'), 'successRateRequestVolume': self.request.get('successRateRequestVolume'), 'successRateStdevFactor': self.request.get('successRateStdevFactor')})