import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.test.secrets import ECS_PARAMS
from libcloud.compute.types import NodeState, StorageVolumeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ecs import ECSDriver
def _create_public_ip_AllocatePublicIpAddress(self, method, url, body, headers):
    resp_body = self.fixtures.load('create_public_ip.xml')
    return (httplib.OK, resp_body, {}, httplib.responses[httplib.OK])