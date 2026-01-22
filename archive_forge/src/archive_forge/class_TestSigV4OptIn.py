import copy
import pickle
import os
from tests.compat import unittest, mock
from tests.unit import MockServiceWithConfigTestCase
from nose.tools import assert_equal
from boto.auth import HmacAuthV4Handler
from boto.auth import S3HmacAuthV4Handler
from boto.auth import detect_potential_s3sigv4
from boto.auth import detect_potential_sigv4
from boto.connection import HTTPRequest
from boto.provider import Provider
from boto.regioninfo import RegionInfo
class TestSigV4OptIn(MockServiceWithConfigTestCase):
    connection_class = FakeEC2Connection

    def setUp(self):
        super(TestSigV4OptIn, self).setUp()
        self.standard_region = RegionInfo(name='us-west-2', endpoint='ec2.us-west-2.amazonaws.com')
        self.sigv4_region = RegionInfo(name='cn-north-1', endpoint='ec2.cn-north-1.amazonaws.com.cn')

    def test_sigv4_opt_out(self):
        fake = FakeEC2Connection(region=self.standard_region)
        self.assertEqual(fake._required_auth_capability(), ['nope'])

    def test_sigv4_non_optional(self):
        fake = FakeEC2Connection(region=self.sigv4_region)
        self.assertEqual(fake._required_auth_capability(), ['hmac-v4'])

    def test_sigv4_opt_in_config(self):
        self.config = {'ec2': {'use-sigv4': True}}
        fake = FakeEC2Connection(region=self.standard_region)
        self.assertEqual(fake._required_auth_capability(), ['hmac-v4'])

    def test_sigv4_opt_in_env(self):
        self.environ['EC2_USE_SIGV4'] = True
        fake = FakeEC2Connection(region=self.standard_region)
        self.assertEqual(fake._required_auth_capability(), ['hmac-v4'])