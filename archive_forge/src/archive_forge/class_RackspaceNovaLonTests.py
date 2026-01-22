import sys
import unittest
from libcloud.pricing import clear_pricing_data
from libcloud.utils.py3 import httplib, method_type
from libcloud.test.secrets import RACKSPACE_PARAMS, RACKSPACE_NOVA_PARAMS
from libcloud.compute.types import Provider
from libcloud.compute.providers import get_driver
from libcloud.compute.drivers.rackspace import RackspaceNodeDriver, RackspaceFirstGenNodeDriver
from libcloud.test.compute.test_openstack import (
class RackspaceNovaLonTests(BaseRackspaceNovaTestCase, OpenStack_1_1_Tests):
    driver_klass = RackspaceNodeDriver
    driver_type = RackspaceNodeDriver
    driver_args = RACKSPACE_NOVA_PARAMS
    driver_kwargs = {'region': 'lon'}
    conn_class = RackspaceNovaLonMockHttp
    auth_url = 'https://lon.auth.api.example.com'
    expected_endpoint = 'https://lon.servers.api.rackspacecloud.com/v2/1337'