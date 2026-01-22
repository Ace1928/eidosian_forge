import json
from boto.cloudtrail.layer1 import CloudTrailConnection
from tests.unit import AWSMockServiceTestCase
class TestCreateTrail(AWSMockServiceTestCase):
    connection_class = CloudTrailConnection

    def default_body(self):
        return b'\n            {"trail":\n                {\n                    "IncludeGlobalServiceEvents": false,\n                    "Name": "test",\n                    "SnsTopicName": "cloudtrail-1",\n                    "S3BucketName": "cloudtrail-1"\n                }\n            }'

    def test_create(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_trail('test', 'cloudtrail-1', sns_topic_name='cloudtrail-1', include_global_service_events=False)
        self.assertEqual('test', api_response['trail']['Name'])
        self.assertEqual('cloudtrail-1', api_response['trail']['S3BucketName'])
        self.assertEqual('cloudtrail-1', api_response['trail']['SnsTopicName'])
        self.assertEqual(False, api_response['trail']['IncludeGlobalServiceEvents'])
        target = self.actual_request.headers['X-Amz-Target']
        self.assertTrue('CreateTrail' in target)