import os
import sys
import pytest
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.secrets import NTTCIS_PARAMS
from libcloud.common.nttcis import NttCisPool, NttCisVIPNode, NttCisPoolMember, NttCisAPIException
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.nttcis import NttCisLBDriver
def _caas_2_7_8a8f6abc_2745_4d8a_9cbc_8dabe5a7d0e4_networkDomainVip_sslOffloadProfile_b1d3b5a7_75d7_4c44_a2b7_5bfa773dec63(self, method, url, body, headers):
    body = self.fixtures.load('get_ssl_offload_profile.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])