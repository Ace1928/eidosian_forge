import tempfile
import shutil
import os
import socket
from boto.compat import json
from boto.awslambda.layer1 import AWSLambdaConnection
from tests.unit import AWSMockServiceTestCase
from tests.compat import mock
def test_upload_function_unseekable_file_cannot_tell(self):
    mock_file = mock.Mock()
    mock_file.tell.side_effect = IOError
    with self.assertRaises(TypeError):
        self.service_connection.upload_function(function_name='my-function', function_zip=mock_file, role='myrole', handler='myhandler', mode='event', runtime='nodejs')