import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def test_reboot_bare_metal_node(self):
    nodes = self.driver.list_nodes()
    node = nodes[-1]
    self.assertTrue(node.extra['is_bare_metal'])
    response = self.driver.reboot_node(node)
    self.assertTrue(response)