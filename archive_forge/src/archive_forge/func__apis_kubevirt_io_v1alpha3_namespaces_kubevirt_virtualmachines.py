import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.kubevirt import KubeVirtNodeDriver
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
def _apis_kubevirt_io_v1alpha3_namespaces_kubevirt_virtualmachines(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('get_kube_public_vms.json')
    elif method == 'POST':
        pass
    else:
        AssertionError('Unsupported method')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])