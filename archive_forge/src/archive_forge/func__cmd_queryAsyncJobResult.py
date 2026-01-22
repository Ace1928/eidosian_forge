import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlparse, parse_qsl, assertRaisesRegex
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import Provider
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.providers import get_driver
from libcloud.loadbalancer.drivers.cloudstack import CloudStackLBDriver
def _cmd_queryAsyncJobResult(self, jobid):
    fixture = 'queryAsyncJobResult' + '_' + str(jobid) + '.json'
    body, obj = self._load_fixture(fixture)
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])