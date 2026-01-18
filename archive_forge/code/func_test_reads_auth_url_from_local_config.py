from unittest import mock
import webob
from webob import exc
from heat.common import auth_url
from heat.tests import common
def test_reads_auth_url_from_local_config(self):
    req = webob.Request.blank('/tenant_id/')
    self.middleware(req)
    self.assertIn('X-Auth-Url', req.headers)
    self.assertEqual('foobar', req.headers['X-Auth-Url'])