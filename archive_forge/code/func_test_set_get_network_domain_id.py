import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.secrets import DIMENSIONDATA_PARAMS
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.common.dimensiondata import (
from libcloud.loadbalancer.drivers.dimensiondata import DimensionDataLBDriver as DimensionData
def test_set_get_network_domain_id(self):
    self.driver.ex_set_current_network_domain('1234')
    nwd = self.driver.ex_get_current_network_domain()
    self.assertEqual(nwd, '1234')