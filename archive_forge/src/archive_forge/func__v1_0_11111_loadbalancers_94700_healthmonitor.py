import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def _v1_0_11111_loadbalancers_94700_healthmonitor(self, method, url, body, headers):
    if method == 'PUT':
        json_body = json.loads(body)
        self.assertEqual('HTTP', json_body['type'])
        self.assertEqual(10, json_body['delay'])
        self.assertEqual(5, json_body['timeout'])
        self.assertEqual(2, json_body['attemptsBeforeDeactivation'])
        self.assertEqual('/', json_body['path'])
        self.assertEqual('^[234][0-9][0-9]$', json_body['statusRegex'])
        self.assertFalse('bodyRegex' in json_body)
        return (httplib.ACCEPTED, '', {}, httplib.responses[httplib.ACCEPTED])
    raise NotImplementedError