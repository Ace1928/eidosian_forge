import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vsphere import VSphere_REST_NodeDriver
class VSphereMockHttp(MockHttp):
    fixtures = ComputeFileFixtures('vsphere')

    def _rest_com_vmware_cis_session(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('session_token.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _rest_vcenter_vm(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('list_nodes.json')
        elif method == 'POST':
            return
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _rest_vcenter_vm_vm_80(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('node_80.json')
        elif method == 'POST':
            return
        elif method == 'DELETE':
            body = ''
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _rest_vcenter_cluster(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('list_clusters.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _rest_vcenter_host(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('list_hosts.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _rest_appliance_networking_interfaces(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('list_interfaces.json')
        else:
            raise AssertionError('Unsupported method')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _rest_vcenter_vm_vm_80_power_stop(self, method, url, body, headers):
        if method != 'POST':
            raise AssertionError('Unsupported method')
        body = ''
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _rest_vcenter_vm_vm_80_power_start(self, method, url, body, headers):
        if method != 'POST':
            raise AssertionError('Unsupported method')
        body = ''
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _rest_vcenter_vm_vm_80_power_reset(self, method, url, body, headers):
        if method != 'POST':
            raise AssertionError('Unsupported method')
        body = ''
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])