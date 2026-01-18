import os
from unittest import mock
import ssl
import testtools
import threading
from glanceclient import Client
from glanceclient import exc
from glanceclient import v1
from glanceclient import v2
import socketserver
@mock.patch('sys.stderr')
def test_v2_requests_bad_ca(self, __):
    """Test VerifiedHTTPSConnection: absence of SSL key file."""
    port = self.port
    url = 'https://0.0.0.0:%d' % port
    cacert = os.path.join(TEST_VAR_DIR, 'badca.crt')
    try:
        gc = Client('2', url, insecure=False, ssl_compression=False, cacert=cacert)
        gc.images.get('image123')
    except exc.CommunicationError as e:
        if 'invalid path' not in e.message:
            raise