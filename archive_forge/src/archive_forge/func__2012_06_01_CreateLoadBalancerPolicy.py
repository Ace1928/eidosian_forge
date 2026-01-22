import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ELB_PARAMS
from libcloud.loadbalancer.base import Member, Algorithm
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.elb import ElasticLBDriver
def _2012_06_01_CreateLoadBalancerPolicy(self, method, url, body, headers):
    body = self.fixtures.load('create_load_balancer_policy.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])