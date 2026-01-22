from boto.compat import json
from boto.machinelearning.layer1 import MachineLearningConnection
from tests.unit import AWSMockServiceTestCase
class TestMachineLearning(AWSMockServiceTestCase):
    connection_class = MachineLearningConnection

    def test_predict(self):
        ml_endpoint = 'mymlmodel.amazonaws.com'
        self.set_http_response(status_code=200, body=b'')
        self.service_connection.predict(ml_model_id='foo', record={'Foo': 'bar'}, predict_endpoint=ml_endpoint)
        self.assertEqual(self.actual_request.host, ml_endpoint)

    def test_predict_with_scheme_in_endpoint(self):
        ml_endpoint = 'mymlmodel.amazonaws.com'
        self.set_http_response(status_code=200, body=b'')
        self.service_connection.predict(ml_model_id='foo', record={'Foo': 'bar'}, predict_endpoint='https://' + ml_endpoint)
        self.assertEqual(self.actual_request.host, ml_endpoint)