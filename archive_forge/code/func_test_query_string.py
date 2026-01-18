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
def test_query_string(self):
    auth = HmacAuthV4Handler('sns.us-east-1.amazonaws.com', mock.Mock(), self.provider)
    params = {'Message': u'We ♥ utf-8'.encode('utf-8')}
    request = HTTPRequest('POST', 'https', 'sns.us-east-1.amazonaws.com', 443, '/', None, params, {}, '')
    query_string = auth.query_string(request)
    self.assertEqual(query_string, 'Message=We%20%E2%99%A5%20utf-8')