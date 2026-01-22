import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def _v1_0_11111_loadbalancers_3133(self, method, url, body, headers):
    """update_balancer(b, algorithm='ROUND_ROBIN'), then get_balancer('3133')"""
    if method == 'PUT':
        json_body = json.loads(body)
        self.assertDictEqual(json_body, {'algorithm': 'ROUND_ROBIN'})
        return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
    elif method == 'GET':
        response_body = json.loads(self.fixtures.load('v1_slug_loadbalancers_3xxx.json'))
        response_body['loadBalancer']['id'] = 3133
        response_body['loadBalancer']['algorithm'] = 'ROUND_ROBIN'
        return (httplib.OK, json.dumps(response_body), {}, httplib.responses[httplib.OK])
    raise NotImplementedError