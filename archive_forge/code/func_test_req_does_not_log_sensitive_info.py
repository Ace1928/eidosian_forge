import json
import logging
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import adapter
from keystoneauth1 import exceptions as keystone_exception
from oslo_serialization import jsonutils
from cinderclient import api_versions
import cinderclient.client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def test_req_does_not_log_sensitive_info(self):
    self.logger = self.useFixture(fixtures.FakeLogger(format='%(message)s', level=logging.DEBUG, nuke_handlers=True))
    secret_auth_token = 'MY_SECRET_AUTH_TOKEN'
    kwargs = {'headers': {'X-Auth-Token': secret_auth_token}, 'data': '{"auth": {"tenantName": "fakeService", "passwordCredentials": {"username": "fakeUser", "password": "fakePassword"}}}'}
    cs = cinderclient.client.HTTPClient('user', None, None, 'http://127.0.0.1:5000')
    cs.http_log_debug = True
    cs.http_log_req('PUT', kwargs)
    output = self.logger.output.split('\n')
    self.assertNotIn(secret_auth_token, output[1])