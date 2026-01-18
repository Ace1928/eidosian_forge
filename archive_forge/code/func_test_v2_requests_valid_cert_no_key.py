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
def test_v2_requests_valid_cert_no_key(self, __):
    """Test VerifiedHTTPSConnection: absence of SSL key file."""
    port = self.port
    url = 'https://0.0.0.0:%d' % port
    cert_file = os.path.join(TEST_VAR_DIR, 'certificate.crt')
    cacert = os.path.join(TEST_VAR_DIR, 'ca.crt')
    try:
        gc = Client('2', url, insecure=False, ssl_compression=False, cert_file=cert_file, cacert=cacert)
        gc.images.get('image123')
    except exc.CommunicationError as e:
        if 'PEM lib' not in e.message:
            self.fail('No appropriate failure message is received')