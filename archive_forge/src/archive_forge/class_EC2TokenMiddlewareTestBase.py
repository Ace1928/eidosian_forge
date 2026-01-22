from unittest import mock
from oslo_serialization import jsonutils
import requests
import webob
from keystonemiddleware import ec2_token
from keystonemiddleware.tests.unit import utils
class EC2TokenMiddlewareTestBase(utils.TestCase):
    TEST_PROTOCOL = 'https'
    TEST_HOST = 'fakehost'
    TEST_PORT = 35357
    TEST_URL = '%s://%s:%d/v3/ec2tokens' % (TEST_PROTOCOL, TEST_HOST, TEST_PORT)

    def setUp(self):
        super(EC2TokenMiddlewareTestBase, self).setUp()
        self.middleware = ec2_token.EC2Token(FakeApp(), {})

    def _validate_ec2_error(self, response, http_status, ec2_code):
        self.assertEqual(http_status, response.status_code, 'Expected HTTP status %s' % http_status)
        error_msg = '<Code>%s</Code>' % ec2_code
        error_msg = error_msg.encode()
        self.assertIn(error_msg, response.body)