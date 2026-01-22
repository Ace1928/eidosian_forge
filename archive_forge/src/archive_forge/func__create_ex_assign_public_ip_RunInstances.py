import os
import sys
import base64
from datetime import datetime
from collections import OrderedDict
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import b, httplib, parse_qs
from libcloud.compute.base import (
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import EC2_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ec2 import (
def _create_ex_assign_public_ip_RunInstances(self, method, url, body, headers):
    self.assertUrlContainsQueryParams(url, {'NetworkInterface.1.AssociatePublicIpAddress': 'true', 'NetworkInterface.1.DeleteOnTermination': 'true', 'NetworkInterface.1.DeviceIndex': '0', 'NetworkInterface.1.SubnetId': 'subnet-11111111', 'NetworkInterface.1.SecurityGroupId.1': 'sg-11111111'})
    body = self.fixtures.load('run_instances_with_subnet_and_security_group.xml')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])