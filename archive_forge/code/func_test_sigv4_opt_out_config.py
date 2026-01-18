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
def test_sigv4_opt_out_config(self):
    self.config = {'s3': {'use-sigv4': 'False'}}
    fake = FakeS3Connection()
    self.assertEqual(fake._required_auth_capability(), ['nope'])