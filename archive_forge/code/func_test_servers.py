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
def test_servers(self):
    mock_uri = 'https://compute.example.com/v2.1/servers/detail'
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=mock_uri, status_code=200, json={'servers': []})])
    list(self.cloud.compute.servers())
    self.assert_calls()
    self.assert_reported_stat('openstack.api.compute.GET.servers_detail.200', value='1', kind='c')
    self.assert_reported_stat('openstack.api.compute.GET.servers_detail.200', value='0', kind='ms')
    self.assert_prometheus_stat('openstack_http_requests_total', 1, dict(service_type='compute', endpoint=mock_uri, method='GET', status_code='200'))