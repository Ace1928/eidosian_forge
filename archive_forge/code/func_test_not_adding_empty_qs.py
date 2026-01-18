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
def test_not_adding_empty_qs(self):
    self.provider.security_token = None
    auth = HmacAuthV4Handler('glacier.us-east-1.amazonaws.com', mock.Mock(), self.provider)
    req = copy.copy(self.request)
    auth.add_auth(req)
    self.assertEqual(req.path, '/-/vaults/foo/archives')