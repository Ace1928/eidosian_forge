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
def _CreateLoadBalancer(self, method, url, body, headers):
    params = {'RegionId': self.test.region, 'LoadBalancerName': self.test.name}
    balancer_keys = {'AddressType': 'ex_address_type', 'InternetChargeType': 'ex_internet_charge_type', 'Bandwidth': 'ex_bandwidth', 'MasterZoneId': 'ex_master_zone_id', 'SlaveZoneId': 'ex_slave_zone_id'}
    for key in balancer_keys:
        params[key] = str(self.test.extra[balancer_keys[key]])
    self.assertUrlContainsQueryParams(url, params)
    body = self.fixtures.load('create_load_balancer.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])