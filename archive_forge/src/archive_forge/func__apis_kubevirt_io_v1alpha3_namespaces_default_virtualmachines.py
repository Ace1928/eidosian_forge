import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.kubevirt import KubeVirtNodeDriver
from libcloud.test.common.test_kubernetes import KubernetesAuthTestCaseMixin
def _apis_kubevirt_io_v1alpha3_namespaces_default_virtualmachines(self, method, url, body, headers):
    if method == 'GET':
        body = self.fixtures.load('get_default_vms.json')
        resp = httplib.OK
    elif method == 'POST':
        body = self.fixtures.load('create_vm.json')
        resp = httplib.CREATED
    else:
        AssertionError('Unsupported method')
    return (resp, body, {}, httplib.responses[httplib.OK])