import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vsphere import VSphere_REST_NodeDriver
def _rest_vcenter_vm_vm_80_power_stop(self, method, url, body, headers):
    if method != 'POST':
        raise AssertionError('Unsupported method')
    body = ''
    return (httplib.OK, body, {}, httplib.responses[httplib.OK])