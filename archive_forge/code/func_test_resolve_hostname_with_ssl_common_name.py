import mock
import os
import json
from nose.tools import assert_equal
from tests.unit import unittest
import boto
from boto.endpoints import BotoEndpointResolver
from boto.endpoints import StaticEndpointBuilder
def test_resolve_hostname_with_ssl_common_name(self):
    resolver = BotoEndpointResolver(self._endpoint_data())
    hostname = resolver.resolve_hostname('s3', 'us-foo')
    expected_hostname = 'us-foo.s3.amazonaws.com'
    self.assertEqual(hostname, expected_hostname)