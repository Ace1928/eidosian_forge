import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def _v1_0_11111_loadbalancers_EX_MEMBER_ADDRESS(self, method, url, body, headers):
    body = self.fixtures.load('v1_slug_loadbalancers_nodeaddress.json')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])