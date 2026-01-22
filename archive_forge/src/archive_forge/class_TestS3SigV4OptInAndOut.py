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
class TestS3SigV4OptInAndOut(MockServiceWithConfigTestCase):
    connection_class = FakeS3Connection

    def test_sigv4_opt_in_config(self):
        self.config = {'s3': {'use-sigv4': 'true'}}
        fake = FakeS3Connection()
        self.assertEqual(fake._required_auth_capability(), ['hmac-v4-s3'])

    def test_sigv4_opt_out_config(self):
        self.config = {'s3': {'use-sigv4': 'False'}}
        fake = FakeS3Connection()
        self.assertEqual(fake._required_auth_capability(), ['nope'])

    def test_sigv4_incorrect_config(self):
        """Test that default(sigv4) is chosen if incorrect value is present."""
        self.config = {'s3': {'use-sigv4': 'someval'}}
        fake = FakeS3Connection(host='s3.amazonaws.com')
        self.assertEqual(fake._required_auth_capability(), ['hmac-v4-s3'])

    def test_sigv4_opt_in_env(self):
        self.environ['S3_USE_SIGV4'] = 'True'
        fake = FakeS3Connection(host='s3.amazonaws.com')
        self.assertEqual(fake._required_auth_capability(), ['hmac-v4-s3'])

    def test_sigv4_opt_out_env(self):
        self.environ['S3_USE_SIGV4'] = 'False'
        fake = FakeS3Connection(host='s3.amazonaws.com')
        self.assertEqual(fake._required_auth_capability(), ['nope'])