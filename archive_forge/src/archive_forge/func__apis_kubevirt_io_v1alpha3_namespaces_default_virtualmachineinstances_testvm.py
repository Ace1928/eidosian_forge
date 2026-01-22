import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.kubevirt import KubeVirtNodeDriver
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
def _apis_kubevirt_io_v1alpha3_namespaces_default_virtualmachineinstances_testvm(self, method, url, body, headers):
    if method == 'DELETE':
        body = self.fixtures.load('delete_vmi_testvm.json')
    else:
        AssertionError('Unsupported method')
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])