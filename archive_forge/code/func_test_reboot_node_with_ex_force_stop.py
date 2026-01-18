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
def test_reboot_node_with_ex_force_stop(self):
    ECSMockHttp.type = 'reboot_node_force_stop'
    result = self.driver.reboot_node(self.fake_node, ex_force_stop=True)
    self.assertTrue(result)