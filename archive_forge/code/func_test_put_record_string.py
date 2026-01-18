from boto.compat import json
from boto.kinesis.layer1 import KinesisConnection
from tests.unit import AWSMockServiceTestCase
def test_put_record_string(self):
    self.set_http_response(status_code=200)
    self.service_connection.put_record('stream-name', 'data', 'partition-key')
    body = json.loads(self.actual_request.body.decode('utf-8'))
    self.assertEqual(body['Data'], 'ZGF0YQ==')
    target = self.actual_request.headers['X-Amz-Target']
    self.assertTrue('PutRecord' in target)