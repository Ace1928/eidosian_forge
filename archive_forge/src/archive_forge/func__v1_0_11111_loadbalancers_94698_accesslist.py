import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def _v1_0_11111_loadbalancers_94698_accesslist(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('v1_slug_loadbalancers_94698_accesslist.json')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])
    elif method == 'POST':
        json_body = json.loads(body)
        self.assertEqual('0.0.0.0/0', json_body['networkItem']['address'])
        self.assertEqual('DENY', json_body['networkItem']['type'])
        return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
    raise NotImplementedError