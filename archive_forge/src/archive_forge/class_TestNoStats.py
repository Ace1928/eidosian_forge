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
class TestNoStats(base.TestCase):

    def setUp(self):
        super(TestNoStats, self).setUp()
        self.statsd = StatsdFixture()
        self.useFixture(self.statsd)

    def test_no_stats(self):
        mock_uri = self.get_mock_url(service_type='identity', resource='projects', base_url_append='v3')
        self.register_uris([dict(method='GET', uri=mock_uri, status_code=200, json={'projects': []})])
        self.cloud.identity._statsd_client = None
        list(self.cloud.identity.projects())
        self.assert_calls()
        self.assertEqual([], self.statsd.stats)