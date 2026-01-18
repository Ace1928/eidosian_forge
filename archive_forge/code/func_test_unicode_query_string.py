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
def test_unicode_query_string(self):
    request = HTTPRequest(method='HEAD', protocol='https', host='awesome-bucket.s3-us-west-2.amazonaws.com', port=443, path=u'/?max-keys=1&prefix=El%20Ni%C3%B1o', auth_path=u'/awesome-bucket/?max-keys=1&prefix=El%20Ni%C3%B1o', params={}, headers={}, body='')
    mod_req = self.auth.mangle_path_and_params(request)
    self.assertEqual(mod_req.path, u'/?max-keys=1&prefix=El%20Ni%C3%B1o')
    self.assertEqual(mod_req.auth_path, u'/awesome-bucket/')
    self.assertEqual(mod_req.params, {u'max-keys': u'1', u'prefix': u'El Niño'})