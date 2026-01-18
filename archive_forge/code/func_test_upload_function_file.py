import tempfile
import shutil
import os
import socket
from boto.compat import json
from boto.awslambda.layer1 import AWSLambdaConnection
from tests.unit import AWSMockServiceTestCase
from tests.compat import mock
def test_upload_function_file(self):
    self.set_http_response(status_code=201)
    rootdir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, rootdir)
    filename = 'test_file'
    function_data = b'This is my file'
    full_path = os.path.join(rootdir, filename)
    with open(full_path, 'wb') as f:
        f.write(function_data)
    with open(full_path, 'rb') as f:
        self.service_connection.upload_function(function_name='my-function', function_zip=f, role='myrole', handler='myhandler', mode='event', runtime='nodejs')
        self.assertEqual(self.actual_request.body.read(), function_data)
        self.assertEqual(self.actual_request.headers['Content-Length'], str(len(function_data)))
        self.assertEqual(self.actual_request.path, '/2014-11-13/functions/my-function?Handler=myhandler&Mode=event&Role=myrole&Runtime=nodejs')