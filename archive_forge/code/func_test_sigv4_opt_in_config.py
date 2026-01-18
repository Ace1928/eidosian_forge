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
def test_sigv4_opt_in_config(self):
    self.config = {'ec2': {'use-sigv4': True}}
    fake = FakeEC2Connection(region=self.standard_region)
    self.assertEqual(fake._required_auth_capability(), ['hmac-v4'])