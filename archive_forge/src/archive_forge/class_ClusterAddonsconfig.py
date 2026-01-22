from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import time
class ClusterAddonsconfig(object):

    def __init__(self, request, module):
        self.module = module
        if request:
            self.request = request
        else:
            self.request = {}

    def to_request(self):
        return remove_nones_from_dict({u'httpLoadBalancing': ClusterHttploadbalancing(self.request.get('http_load_balancing', {}), self.module).to_request(), u'horizontalPodAutoscaling': ClusterHorizontalpodautoscaling(self.request.get('horizontal_pod_autoscaling', {}), self.module).to_request(), u'networkPolicyConfig': ClusterNetworkpolicyconfig(self.request.get('network_policy_config', {}), self.module).to_request()})

    def from_response(self):
        return remove_nones_from_dict({u'httpLoadBalancing': ClusterHttploadbalancing(self.request.get(u'httpLoadBalancing', {}), self.module).from_response(), u'horizontalPodAutoscaling': ClusterHorizontalpodautoscaling(self.request.get(u'horizontalPodAutoscaling', {}), self.module).from_response(), u'networkPolicyConfig': ClusterNetworkpolicyconfig(self.request.get(u'networkPolicyConfig', {}), self.module).from_response()})