import sys
import unittest
from libcloud.pricing import clear_pricing_data
from libcloud.utils.py3 import httplib, method_type
from libcloud.test.secrets import RACKSPACE_PARAMS, RACKSPACE_NOVA_PARAMS
from libcloud.compute.types import Provider
from libcloud.compute.providers import get_driver
from libcloud.compute.drivers.rackspace import RackspaceNodeDriver, RackspaceFirstGenNodeDriver
from libcloud.test.compute.test_openstack import (
class RackspaceNovaMockHttp(OpenStack_1_1_MockHttp):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        methods1 = OpenStack_1_1_MockHttp.__dict__
        names1 = [m for m in methods1 if m.find('_v1_1') == 0]
        for name in names1:
            method = methods1[name]
            new_name = name.replace('_v1_1_slug_', '_v2_1337_')
            setattr(self, new_name, method_type(method, self, RackspaceNovaMockHttp))

    def _v2_1337_os_networksv2(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('_os_networks.json')
            return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])
        elif method == 'POST':
            body = self.fixtures.load('_os_networks_POST.json')
            return (httplib.ACCEPTED, body, self.json_content_headers, httplib.responses[httplib.OK])
        raise NotImplementedError()

    def _v2_1337_os_networksv2_f13e5051_feea_416b_827a_1a0acc2dad14(self, method, url, body, headers):
        if method == 'DELETE':
            body = ''
            return (httplib.ACCEPTED, body, self.json_content_headers, httplib.responses[httplib.OK])
        raise NotImplementedError()