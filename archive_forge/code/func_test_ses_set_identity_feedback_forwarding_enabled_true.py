from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.jsonresponse import ListElement
from boto.ses.connection import SESConnection
def test_ses_set_identity_feedback_forwarding_enabled_true(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.set_identity_feedback_forwarding_enabled(identity='user@example.com', forwarding_enabled=True)
    response = response['SetIdentityFeedbackForwardingEnabledResponse']
    result = response['SetIdentityFeedbackForwardingEnabledResult']
    self.assertEqual(2, len(response))
    self.assertEqual(0, len(result))