from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.jsonresponse import ListElement
from boto.ses.connection import SESConnection
def test_ses_set_identity_notification_topic_complaint(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.set_identity_notification_topic(identity='user@example.com', notification_type='Complaint', sns_topic='arn:aws:sns:us-east-1:123456789012:example')
    response = response['SetIdentityNotificationTopicResponse']
    result = response['SetIdentityNotificationTopicResult']
    self.assertEqual(2, len(response))
    self.assertEqual(0, len(result))