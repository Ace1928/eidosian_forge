import hashlib
import http.client as http
import os
import subprocess
import tempfile
import time
import urllib
import uuid
import fixtures
from oslo_limit import exception as ol_exc
from oslo_limit import limit
from oslo_serialization import jsonutils
from oslo_utils.secretutils import md5
from oslo_utils import units
import requests
from glance.quota import keystone as ks_quota
from glance.tests import functional
from glance.tests.functional import ft_utils as func_utils
from glance.tests import utils as test_utils
class TestImagesIPv6(functional.FunctionalTest):
    """Verify that API and REG servers running IPv6 can communicate"""

    def setUp(self):
        """
        First applying monkey patches of functions and methods which have
        IPv4 hardcoded.
        """
        test_utils.get_unused_port_ipv4 = test_utils.get_unused_port
        test_utils.get_unused_port_and_socket_ipv4 = test_utils.get_unused_port_and_socket
        test_utils.get_unused_port = test_utils.get_unused_port_ipv6
        test_utils.get_unused_port_and_socket = test_utils.get_unused_port_and_socket_ipv6
        super(TestImagesIPv6, self).setUp()
        self.cleanup()
        self.ping_server_ipv4 = self.ping_server
        self.ping_server = self.ping_server_ipv6
        self.include_scrubber = False

    def tearDown(self):
        self.ping_server = self.ping_server_ipv4
        super(TestImagesIPv6, self).tearDown()
        test_utils.get_unused_port = test_utils.get_unused_port_ipv4
        test_utils.get_unused_port_and_socket = test_utils.get_unused_port_and_socket_ipv4

    def _headers(self, custom_headers=None):
        base_headers = {'X-Identity-Status': 'Confirmed', 'X-Auth-Token': '932c5c84-02ac-4fe5-a9ba-620af0e2bb96', 'X-User-Id': 'f9a41d13-0c13-47e9-bee2-ce4e8bfe958e', 'X-Tenant-Id': TENANT1, 'X-Roles': 'reader,member'}
        base_headers.update(custom_headers or {})
        return base_headers

    def test_image_list_ipv6(self):
        self.api_server.deployment_flavor = 'caching'
        self.start_servers(**self.__dict__.copy())
        url = f'http://[::1]:{self.api_port}'
        path = '/'
        requests.get(url + path, headers=self._headers())
        path = '/v2/images'
        response = requests.get(url + path, headers=self._headers())
        self.assertEqual(200, response.status_code)
        images = jsonutils.loads(response.text)['images']
        self.assertEqual(0, len(images))