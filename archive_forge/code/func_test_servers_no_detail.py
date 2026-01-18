import itertools
import os
import pprint
import select
import socket
import threading
import time
import fixtures
from keystoneauth1 import exceptions
import prometheus_client
from requests import exceptions as rexceptions
import testtools.content
from openstack.tests.unit import base
def test_servers_no_detail(self):
    mock_uri = 'https://compute.example.com/v2.1/servers'
    self.register_uris([dict(method='GET', uri=mock_uri, status_code=200, json={'servers': []})])
    self.cloud.compute.get('/servers')
    self.assert_calls()
    self.assert_reported_stat('openstack.api.compute.GET.servers.200', value='1', kind='c')
    self.assert_reported_stat('openstack.api.compute.GET.servers.200', value='0', kind='ms')
    self.assert_reported_stat('openstack.api.compute.GET.servers.attempted', value='1', kind='c')
    self.assert_prometheus_stat('openstack_http_requests_total', 1, dict(service_type='compute', endpoint=mock_uri, method='GET', status_code='200'))