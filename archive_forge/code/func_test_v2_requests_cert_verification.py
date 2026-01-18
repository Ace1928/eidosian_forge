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
def test_v2_requests_cert_verification(self, __):
    """v2 regression test for bug 115260."""
    port = self.port
    url = 'https://0.0.0.0:%d' % port
    try:
        gc = v2.Client(url, insecure=False, ssl_compression=True)
        gc.images.get('image123')
        self.fail('No SSL exception has been raised')
    except exc.CommunicationError as e:
        if 'certificate verify failed' not in e.message:
            self.fail('No certificate failure message is received')