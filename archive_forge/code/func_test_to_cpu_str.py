import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import CONTAINER_PARAMS_KUBERNETES
from libcloud.container.base import ContainerImage
from libcloud.test.file_fixtures import ContainerFileFixtures
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
from libcloud.container.drivers.kubernetes import (
def test_to_cpu_str(self):
    cpu = 0
    self.assertEqual(to_cpu_str(cpu), '0')
    cpu = 0.5
    self.assertEqual(to_cpu_str(cpu), '500m')
    cpu = 2
    self.assertEqual(to_cpu_str(cpu), '2000m')
    cpu = 1e-06
    self.assertEqual(to_cpu_str(cpu), '1u')
    cpu = 0.0005
    self.assertEqual(to_cpu_str(cpu), '500u')
    cpu = 1e-09
    self.assertEqual(to_cpu_str(cpu), '1n')
    cpu = 5e-07
    self.assertEqual(to_cpu_str(cpu), '500n')