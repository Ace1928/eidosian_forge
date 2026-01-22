import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ALB_PARAMS
from libcloud.loadbalancer.base import Member
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.alb import ApplicationLBDriver
def _2015_12_01_CreateRule(self, method, url, body, headers):
    body = self.fixtures.load('create_rule.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])