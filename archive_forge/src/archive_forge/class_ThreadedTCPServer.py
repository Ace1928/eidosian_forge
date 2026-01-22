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
class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):

    def get_request(self):
        key_file = os.path.join(TEST_VAR_DIR, 'privatekey.key')
        cert_file = os.path.join(TEST_VAR_DIR, 'certificate.crt')
        cacert = os.path.join(TEST_VAR_DIR, 'ca.crt')
        _sock, addr = socketserver.TCPServer.get_request(self)
        sock = ssl.wrap_socket(_sock, certfile=cert_file, keyfile=key_file, ca_certs=cacert, server_side=True, cert_reqs=ssl.CERT_REQUIRED)
        return (sock, addr)