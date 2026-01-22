from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class RegionBackendServiceBackendsArray(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = []

    def to_request(self):
        items = []
        for item in self.request:
            items.append(self._request_for_item(item))
        return items

    def from_response(self):
        items = []
        for item in self.request:
            items.append(self._response_from_item(item))
        return items

    def _request_for_item(self, item):
        return remove_nones_from_dict({u'balancingMode': item.get('balancing_mode'), u'capacityScaler': item.get('capacity_scaler'), u'description': item.get('description'), u'failover': item.get('failover'), u'group': item.get('group'), u'maxConnections': item.get('max_connections'), u'maxConnectionsPerInstance': item.get('max_connections_per_instance'), u'maxConnectionsPerEndpoint': item.get('max_connections_per_endpoint'), u'maxRate': item.get('max_rate'), u'maxRatePerInstance': item.get('max_rate_per_instance'), u'maxRatePerEndpoint': item.get('max_rate_per_endpoint'), u'maxUtilization': item.get('max_utilization')})

    def _response_from_item(self, item):
        return remove_nones_from_dict({u'balancingMode': item.get(u'balancingMode'), u'capacityScaler': item.get(u'capacityScaler'), u'description': item.get(u'description'), u'failover': item.get(u'failover'), u'group': item.get(u'group'), u'maxConnections': item.get(u'maxConnections'), u'maxConnectionsPerInstance': item.get(u'maxConnectionsPerInstance'), u'maxConnectionsPerEndpoint': item.get(u'maxConnectionsPerEndpoint'), u'maxRate': item.get(u'maxRate'), u'maxRatePerInstance': item.get(u'maxRatePerInstance'), u'maxRatePerEndpoint': item.get(u'maxRatePerEndpoint'), u'maxUtilization': item.get(u'maxUtilization')})