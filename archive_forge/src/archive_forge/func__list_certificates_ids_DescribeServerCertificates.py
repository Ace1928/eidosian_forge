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
def _list_certificates_ids_DescribeServerCertificates(self, method, url, body, headers):
    params = {'RegionId': self.test.region, 'ServerCertificateId': ','.join(self.test.cert_ids)}
    self.assertUrlContainsQueryParams(url, params)
    body = self.fixtures.load('describe_server_certificates.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])