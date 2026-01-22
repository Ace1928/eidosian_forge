import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node
from libcloud.test.secrets import LB_SLB_PARAMS
from libcloud.compute.types import NodeState
from libcloud.loadbalancer.base import Member, Algorithm
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.slb import (
def _create_listener_override_port_CreateLoadBalancerHTTPListener(self, method, url, body, headers):
    params = {'LoadBalancerId': self.test.balancer.id, 'RegionId': self.test.region, 'ListenerPort': str(self.test.extra['ListenerPort']), 'BackendServerPort': str(self.test.backend_port), 'Scheduler': 'wrr', 'Bandwidth': '1', 'StickySession': 'off', 'HealthCheck': 'off'}
    self.assertUrlContainsQueryParams(url, params)
    body = self.fixtures.load('create_load_balancer_http_listener.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])