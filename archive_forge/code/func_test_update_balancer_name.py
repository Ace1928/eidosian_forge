import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_update_balancer_name(self):
    balancer = LoadBalancer(id='3132', name='LB_update', state='PENDING_UPDATE', ip='10.34.4.3', port=80, driver=self.driver)
    updated_balancer = self.driver.update_balancer(balancer, name='new_lb_name')
    self.assertEqual('new_lb_name', updated_balancer.name)