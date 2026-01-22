import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.kubevirt import KubeVirtNodeDriver
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
def _api_v1_namespaces_default_services(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('get_services.json')
    else:
        AssertionError('Unsupported method')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])