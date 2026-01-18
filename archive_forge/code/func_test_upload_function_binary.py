import tempfile
import shutil
import os
import socket
from boto.compat import json
from boto.awslambda.layer1 import AWSLambdaConnection
from tests.unit import AWSMockServiceTestCase
from tests.compat import mock
def test_upload_function_binary(self):
    self.set_http_response(status_code=201)
    function_data = b'This is my file'
    self.service_connection.upload_function(function_name='my-function', function_zip=function_data, role='myrole', handler='myhandler', mode='event', runtime='nodejs')
    self.assertEqual(self.actual_request.body, function_data)
    self.assertEqual(self.actual_request.headers['Content-Length'], str(len(function_data)))
    self.assertEqual(self.actual_request.path, '/2014-11-13/functions/my-function?Handler=myhandler&Mode=event&Role=myrole&Runtime=nodejs')