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
def test_region_stripping(self):
    auth = S3HmacAuthV4Handler(host='s3-us-west-2.amazonaws.com', config=mock.Mock(), provider=self.provider)
    self.assertEqual(auth.region_name, None)
    auth = S3HmacAuthV4Handler(host='s3-us-west-2.amazonaws.com', config=mock.Mock(), provider=self.provider, region_name='us-west-2')
    self.assertEqual(auth.region_name, 'us-west-2')
    self.assertEqual(self.auth.region_name, 'us-west-2')