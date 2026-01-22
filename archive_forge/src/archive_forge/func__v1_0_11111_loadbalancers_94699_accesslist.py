import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def _v1_0_11111_loadbalancers_94699_accesslist(self, method, url, body, headers):
    if method == 'DELETE':
        fixture = 'v1_slug_loadbalancers_94698_with_access_list.json'
        fixture_json = json.loads(self.fixtures.load(fixture))
        access_list_json = fixture_json['loadBalancer']['accessList']
        for access_rule in access_list_json:
            id = access_rule['id']
            self.assertTrue(urlencode([('id', id)]) in url, msg='Did not delete access rule with id %d' % id)
        return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
    elif method == 'POST':
        json_body = json.loads(body)
        access_list = json_body['accessList']
        self.assertEqual('ALLOW', access_list[0]['type'])
        self.assertEqual('2001:4801:7901::6/64', access_list[0]['address'])
        self.assertEqual('DENY', access_list[1]['type'])
        self.assertEqual('8.8.8.8/0', access_list[1]['address'])
        return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
    raise NotImplementedError