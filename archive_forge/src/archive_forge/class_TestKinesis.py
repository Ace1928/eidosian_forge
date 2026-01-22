from boto.compat import json
from boto.kms.layer1 import KMSConnection
from tests.unit import AWSMockServiceTestCase
class TestKinesis(AWSMockServiceTestCase):
    connection_class = KMSConnection

    def default_body(self):
        return b'{}'

    def test_binary_input(self):
        """
        This test ensures that binary is base64 encoded when it is sent to
        the service.
        """
        self.set_http_response(status_code=200)
        data = b'\x00\x01\x02\x03\x04\x05'
        self.service_connection.encrypt(key_id='foo', plaintext=data)
        body = json.loads(self.actual_request.body.decode('utf-8'))
        self.assertEqual(body['Plaintext'], 'AAECAwQF')

    def test_non_binary_input_for_blobs_fails(self):
        """
        This test ensures that only binary is used for blob type parameters.
        """
        self.set_http_response(status_code=200)
        data = u'é'
        with self.assertRaises(TypeError):
            self.service_connection.encrypt(key_id='foo', plaintext=data)

    def test_binary_ouput(self):
        """
        This test ensures that the output is base64 decoded before
        it is returned to the user.
        """
        content = {'Plaintext': 'AAECAwQF'}
        self.set_http_response(status_code=200, body=json.dumps(content).encode('utf-8'))
        response = self.service_connection.decrypt(b'some arbitrary value')
        self.assertEqual(response['Plaintext'], b'\x00\x01\x02\x03\x04\x05')