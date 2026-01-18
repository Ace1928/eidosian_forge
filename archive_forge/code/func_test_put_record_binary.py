from boto.compat import json
from boto.kinesis.layer1 import KinesisConnection
from tests.unit import AWSMockServiceTestCase
def test_put_record_binary(self):
    self.set_http_response(status_code=200)
    self.service_connection.put_record('stream-name', b'\x00\x01\x02\x03\x04\x05', 'partition-key')
    body = json.loads(self.actual_request.body.decode('utf-8'))
    self.assertEqual(body['Data'], 'AAECAwQF')
    target = self.actual_request.headers['X-Amz-Target']
    self.assertTrue('PutRecord' in target)