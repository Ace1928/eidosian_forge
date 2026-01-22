from unittest import mock
from oslo_serialization import jsonutils
import requests
import webob
from keystonemiddleware import ec2_token
from keystonemiddleware.tests.unit import utils
def _validate_ec2_error(self, response, http_status, ec2_code):
    self.assertEqual(http_status, response.status_code, 'Expected HTTP status %s' % http_status)
    error_msg = '<Code>%s</Code>' % ec2_code
    error_msg = error_msg.encode()
    self.assertIn(error_msg, response.body)